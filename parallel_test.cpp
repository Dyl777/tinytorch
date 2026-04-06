#include "parallel.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

using DW = DenseTensorWorker<float>;
using MW = MmapTensorWorker<float>;

void test_shard_descriptor() {
    std::cout << "=== Testing Shard Descriptor ===" << std::endl;
    ShardDescriptor desc(0, 4, 0, 25, {100}, BackendType::CPU_DENSE);
    assert(desc.shard_id == 0);
    assert(desc.total_shards == 4);
    assert(desc.offset == 0);
    assert(desc.count == 25);
    std::cout << "[OK] Basic shard descriptor" << std::endl;

    auto d0 = ShardDescriptor::create_for_rank(0, 4, {100}, BackendType::CPU_DENSE);
    auto d3 = ShardDescriptor::create_for_rank(3, 4, {100}, BackendType::OPENGL);
    assert(d0.offset == 0); assert(d3.offset == 75);
    assert(d0.count == 25); assert(d3.count == 25);
    std::cout << "[OK] Rank-based shard descriptor with backend types" << std::endl;

    auto du = ShardDescriptor::create_for_rank(3, 4, {10}, BackendType::OPENCL);
    assert(du.offset == 9); assert(du.count == 1);
    std::cout << "[OK] Uneven division handling" << std::endl;
    std::cout << "[OK] Shard descriptor tests passed\n" << std::endl;
}

void test_parallel_config() {
    std::cout << "=== Testing Parallel Config ===" << std::endl;
    auto dp = ParallelConfig::data_parallel(4, 2, BackendType::CPU_MMAP);
    assert(dp.dp_rank() == 2); assert(dp.preferred_backend == BackendType::CPU_MMAP);
    std::cout << "[OK] Data parallel config" << std::endl;

    auto tp = ParallelConfig::tensor_parallel(8, 3, BackendType::CUDA);
    assert(tp.tp_rank() == 3); assert(tp.preferred_backend == BackendType::CUDA);
    std::cout << "[OK] Tensor parallel config" << std::endl;

    auto h3 = ParallelConfig::hybrid_3d(4, 2, 3, 17, BackendType::OPENGL);
    assert(h3.world_size == 24); assert(h3.tp_rank() == 1);
    assert(h3.dp_rank() == 0); assert(h3.pp_rank() == 2);
    std::cout << "[OK] Hybrid 3D config" << std::endl;

    auto ep = ParallelConfig::expert_parallel(8, 3, 16, 4, BackendType::OPENCL);
    assert(ep.num_experts == 16); assert(ep.num_redundant_experts == 4);
    std::cout << "[OK] Expert parallel config" << std::endl;
    std::cout << "[OK] Parallel config tests passed\n" << std::endl;
}

void test_tensor_mailbox() {
    std::cout << "=== Testing Tensor Mailbox ===" << std::endl;
    TensorMailbox<float> mailbox;
    float data[] = {1.0f, 2.0f, 3.0f};
    auto tensor = std::make_unique<DenseTensor<float>>(data, 3);

    std::thread sender([&mailbox, &tensor]() { mailbox.send(std::move(tensor)); });
    sender.join();

    std::unique_ptr<TensorBase<float>> received;
    std::thread receiver([&mailbox, &received]() { received = mailbox.recv(); });
    receiver.join();

    assert(received != nullptr);
    assert(received->get_element(0) == 1.0f);
    assert(received->get_element(2) == 3.0f);
    std::cout << "[OK] Tensor mailbox send/recv" << std::endl;

    std::unique_ptr<TensorBase<float>> timeout_result;
    assert(!mailbox.try_recv(timeout_result, 100));
    std::cout << "[OK] Tensor mailbox timeout" << std::endl;
    std::cout << "[OK] Tensor mailbox tests passed\n" << std::endl;
}

void test_dense_tensor_worker() {
    std::cout << "=== Testing Dense Tensor Worker ===" << std::endl;
    ParallelConfig cfg = ParallelConfig::data_parallel(4, 0, BackendType::CPU_DENSE);
    auto worker = std::make_unique<DW>(0, cfg);

    assert(worker->worker_id() == 0);
    assert(worker->backend_type() == BackendType::CPU_DENSE);
    assert(worker->is_healthy());

    auto hb = worker->get_heartbeat();
    assert(hb.worker_id == 0);
    assert(hb.backend == BackendType::CPU_DENSE);
    std::cout << "[OK] Worker creation and heartbeat" << std::endl;

    float weights[] = {2.0f, 2.0f, 2.0f};
    auto model = std::make_unique<DenseTensor<float>>(weights, 3);
    worker->set_model_shard(std::move(model));

    float input[] = {1.0f, 2.0f, 3.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 3);
    auto output = worker->forward_shard(input_tensor.get());

    assert(output != nullptr);
    assert(std::abs(output->get_element(0) - 2.0f) < 1e-5f);
    assert(std::abs(output->get_element(2) - 6.0f) < 1e-5f);
    std::cout << "[OK] Forward pass with model shard" << std::endl;

    float grad_out[] = {1.0f, 1.0f, 1.0f};
    auto grad_tensor = std::make_unique<DenseTensor<float>>(grad_out, 3);
    auto grad = worker->backward_shard(grad_tensor.get());
    assert(grad != nullptr);
    std::cout << "[OK] Backward pass computes gradients" << std::endl;

    worker->sgd_step(0.1f);
    std::cout << "[OK] SGD step executed" << std::endl;
    std::cout << "[OK] Dense tensor worker tests passed\n" << std::endl;
}

void test_mmap_tensor_worker() {
    std::cout << "=== Testing Mmap Tensor Worker ===" << std::endl;
    ParallelConfig cfg = ParallelConfig::data_parallel(4, 0, BackendType::CPU_MMAP);
    auto worker = std::make_unique<MW>(0, cfg);

    assert(worker->worker_id() == 0);
    assert(worker->backend_type() == BackendType::CPU_MMAP);

    auto hb = worker->get_heartbeat();
    assert(hb.backend == BackendType::CPU_MMAP);
    std::cout << "[OK] Mmap worker creation and heartbeat" << std::endl;

    std::size_t shape[] = {3};
    StreamConfig sc; sc.batch_size = 1024;
    auto model = std::make_unique<MmapTensor<float>>(shape, 1, sc, 3.0f);
    worker->set_model_shard(std::move(model));

    float input[] = {1.0f, 2.0f, 3.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 3);
    auto output = worker->forward_shard(input_tensor.get());

    assert(output != nullptr);
    assert(std::abs(output->get_element(0) - 3.0f) < 1e-5f);
    std::cout << "[OK] Mmap forward pass works" << std::endl;
    std::cout << "[OK] Mmap tensor worker tests passed\n" << std::endl;
}

void test_worker_coordinator() {
    std::cout << "=== Testing Worker Coordinator ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i, BackendType::CPU_DENSE);
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 4; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    std::vector<std::thread> threads;
    std::atomic<int> barrier_reached{0};
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&workers, &barrier_reached, i]() {
            workers[i]->barrier();
            barrier_reached.fetch_add(1);
        });
    }
    for (auto& t : threads) t.join();
    assert(barrier_reached.load() == 4);
    std::cout << "[OK] Barrier synchronization works" << std::endl;
    std::cout << "[OK] Worker coordinator tests passed\n" << std::endl;
}

void test_expert_load_balancer_hierarchical() {
    std::cout << "=== Testing Expert Load Balancer (Hierarchical) ===" << std::endl;
    ExpertLoadBalancer balancer(16, 8, 2, 4, 4, 4);
    std::vector<double> loads(16, 1.0);
    loads[0] = 5.0; loads[1] = 5.0; loads[2] = 0.5;
    balancer.update_expert_loads(loads);

    auto assignment = balancer.rebalance_experts();
    for (int i = 0; i < 16; ++i) assert(!assignment[i].empty());
    std::cout << "[OK] All experts assigned" << std::endl;

    std::vector<double> wl(8, 0);
    for (int e = 0; e < 16; ++e)
        for (int w : assignment[e]) wl[w] += loads[e] / assignment[e].size();

    double mx = 0, mn = 1e18;
    for (double l : wl) { mx = std::max(mx, l); mn = std::min(mn, l); }
    std::cout << "Load imbalance ratio: " << (mx / mn) << std::endl;
    assert(mx / mn < 3.0);
    std::cout << "[OK] Hierarchical load balancer tests passed\n" << std::endl;
}

void test_expert_load_balancer_global() {
    std::cout << "=== Testing Expert Load Balancer (Global) ===" << std::endl;
    ExpertLoadBalancer balancer(12, 6, 2, 3, 2, 0);
    std::vector<double> loads = {3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5};
    balancer.update_expert_loads(loads);

    auto assignment = balancer.rebalance_experts();
    for (int i = 0; i < 12; ++i) assert(!assignment[i].empty());
    std::cout << "[OK] All experts assigned (global)" << std::endl;
    std::cout << "[OK] Global load balancer tests passed\n" << std::endl;
}

void test_distributed_load_balancer() {
    std::cout << "=== Testing Distributed Load Balancer ===" << std::endl;
    auto lb = std::make_unique<DistributedLoadBalancer<float>>(0.1, 5.0);

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i, static_cast<BackendType>(i % 3));
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (auto* w : raw) lb->register_worker(w);

    lb->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    for (int i = 0; i < 10; ++i) {
        int w = lb->assign_shard(i);
        assert(w >= 0 && w < 4);
    }
    std::cout << "[OK] Shard assignment works" << std::endl;

    auto healthy = lb->get_healthy_workers();
    assert(healthy.size() == 4);
    std::cout << "[OK] Healthy worker retrieval works" << std::endl;

    lb->stop();
    std::cout << "[OK] Distributed load balancer tests passed\n" << std::endl;
}

void test_load_balancing_strategies() {
    std::cout << "=== Testing Load Balancing Strategies ===" << std::endl;

    // Create workers
    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i);
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 4; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    // Test each strategy
    LoadBalancingStrategy strategies[] = {
        LoadBalancingStrategy::HEARTBEAT_CAPACITY,
        LoadBalancingStrategy::WEIGHTED_ROUND_ROBIN,
        LoadBalancingStrategy::LEAST_CONNECTIONS,
        LoadBalancingStrategy::POWER_OF_TWO_CHOICES,
        LoadBalancingStrategy::CONSISTENT_HASHING,
        LoadBalancingStrategy::PREDICTIVE_LOAD,
        LoadBalancingStrategy::MIN_MAX_FAIRNESS
    };

    const char* names[] = {
        "Heartbeat Capacity",
        "Weighted Round Robin",
        "Least Connections",
        "Power of Two Choices",
        "Consistent Hashing",
        "Predictive Load",
        "Min-Max Fairness"
    };

    for (int s = 0; s < 7; ++s) {
        auto lb = std::make_unique<DistributedLoadBalancer<float>>(0.05, 5.0, strategies[s]);

        for (auto* w : raw) lb->register_worker(w);
        lb->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Assign 20 shards
        std::vector<int> counts(4, 0);
        for (int i = 0; i < 20; ++i) {
            int w = lb->assign_shard(i);
            assert(w >= 0 && w < 4);
            counts[w]++;
        }

        // Verify all shards were assigned
        int total = 0;
        for (int c : counts) total += c;
        assert(total == 20);

        std::cout << "  " << names[s] << ": [";
        for (int i = 0; i < 4; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << counts[i];
        }
        std::cout << "]" << std::endl;

        lb->stop();
    }

    // Test runtime strategy switching
    auto lb = std::make_unique<DistributedLoadBalancer<float>>(0.05, 5.0, LoadBalancingStrategy::HEARTBEAT_CAPACITY);
    for (auto* w : raw) lb->register_worker(w);
    lb->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Assign with default strategy
    for (int i = 0; i < 5; ++i) lb->assign_shard(i);

    // Switch to Power of Two Choices
    lb->set_strategy(LoadBalancingStrategy::POWER_OF_TWO_CHOICES);
    for (int i = 5; i < 10; ++i) lb->assign_shard(i);

    // Switch to Consistent Hashing
    lb->set_strategy(LoadBalancingStrategy::CONSISTENT_HASHING);
    for (int i = 10; i < 15; ++i) lb->assign_shard(i);

    std::cout << "[OK] Runtime strategy switching works" << std::endl;
    lb->stop();

    std::cout << "[OK] Load balancing strategy tests passed\n" << std::endl;
}

void test_data_parallel_executor() {
    std::cout << "=== Testing Data Parallel Executor ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i, static_cast<BackendType>(i % 3));
        auto w = std::make_unique<DW>(i, cfg);
        float weights[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
        w->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 4; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    DataParallelExecutor<float> executor(dist, &coord);

    std::vector<std::unique_ptr<TensorBase<float>>> shards;
    std::vector<const TensorBase<float>*> ptrs;
    for (int i = 0; i < 4; ++i) {
        float data[] = {static_cast<float>(i*3+1), static_cast<float>(i*3+2), static_cast<float>(i*3+3)};
        shards.push_back(std::make_unique<DenseTensor<float>>(data, 3));
        ptrs.push_back(shards.back().get());
    }

    auto results = executor.forward_all(ptrs);
    assert(results.size() == 4);
    for (int i = 0; i < 4; ++i) {
        float expected = static_cast<float>(i*3+1) * static_cast<float>(i+1);
        assert(std::abs(results[i]->get_element(0) - expected) < 1e-5f);
    }
    std::cout << "[OK] Data parallel forward computes correctly" << std::endl;

    executor.backward_all(ptrs);
    std::cout << "[OK] Data parallel backward executes" << std::endl;

    executor.step_all(0.01f);
    std::cout << "[OK] Data parallel SGD step executes" << std::endl;
    std::cout << "[OK] Data parallel executor tests passed\n" << std::endl;
}

void test_shard_operations() {
    std::cout << "=== Testing Shard Operations ===" << std::endl;
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto full = std::make_unique<DenseTensor<float>>(data, 8);

    std::vector<std::unique_ptr<TensorBase<float>>> shards;
    std::vector<ShardDescriptor> descs;
    for (int i = 0; i < 4; ++i) {
        auto d = ShardDescriptor::create_for_rank(i, 4, {8});
        descs.push_back(d);
        shards.push_back(extract_shard(full.get(), d));
    }

    assert(shards[0]->get_element(0) == 1.0f);
    assert(shards[3]->get_element(0) == 7.0f);
    std::cout << "[OK] Shard extraction works" << std::endl;

    std::vector<const TensorBase<float>*> ptrs;
    for (const auto& s : shards) ptrs.push_back(s.get());
    auto reassembled = assemble_shards(ptrs, descs);
    for (std::size_t i = 0; i < 8; ++i)
        assert(std::abs(reassembled->get_element(i) - data[i]) < 1e-5f);
    std::cout << "[OK] Shard assembly works" << std::endl;
    std::cout << "[OK] Shard operation tests passed\n" << std::endl;
}

void test_tensor_parallel_executor() {
    std::cout << "=== Testing Tensor Parallel Executor ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::tensor_parallel(4, i);
        auto w = std::make_unique<DW>(i, cfg);
        float weights[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
        w->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 4; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    TensorParallelExecutor<float> tp(dist, 4, &coord);

    float input[] = {1.0f, 2.0f, 3.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 3);

    auto col = tp.column_parallel_forward(input_tensor.get());
    assert(col != nullptr); assert(col->total_size() == 12);
    std::cout << "[OK] Column parallel forward" << std::endl;

    auto row = tp.row_parallel_forward(input_tensor.get());
    assert(row != nullptr); assert(row->total_size() == 3);
    std::cout << "[OK] Row parallel forward" << std::endl;
    std::cout << "[OK] Tensor parallel executor tests passed\n" << std::endl;
}

void test_pipeline_parallel_executor() {
    std::cout << "=== Testing Pipeline Parallel Executor ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 3; ++i) {
        auto cfg = ParallelConfig::pipeline_parallel(3, i);
        auto w = std::make_unique<DW>(i, cfg);
        float weights[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
        w->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 3; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    PipelineParallelExecutor<float> pp(dist, 3, 2);

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 6);
    auto output = pp.forward_pipeline(input_tensor.get());
    assert(output != nullptr);
    std::cout << "[OK] Pipeline forward produces output" << std::endl;

    float grad[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto grad_tensor = std::make_unique<DenseTensor<float>>(grad, 6);
    pp.backward_pipeline(grad_tensor.get());
    std::cout << "[OK] Pipeline backward executes" << std::endl;
    std::cout << "[OK] Pipeline parallel executor tests passed\n" << std::endl;
}

void test_fsdp_executor() {
    std::cout << "=== Testing FSDP Executor ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::fsdp(4, i);
        auto w = std::make_unique<DW>(i, cfg);
        float weights[] = {static_cast<float>(i*3+1), static_cast<float>(i*3+2), static_cast<float>(i*3+3)};
        w->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }
    for (int i = 0; i < 4; ++i) workers[i]->set_peers(raw);

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    FSDPExecutor<float> fsdp(dist, 4, &coord);

    std::vector<std::unique_ptr<TensorBase<float>>> dshards;
    std::vector<const TensorBase<float>*> ptrs;
    for (int i = 0; i < 4; ++i) {
        float data[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
        dshards.push_back(std::make_unique<DenseTensor<float>>(data, 3));
        ptrs.push_back(dshards.back().get());
    }
    float pdata[] = {1.0f, 2.0f, 3.0f};
    auto pshard = std::make_unique<DenseTensor<float>>(pdata, 3);

    auto results = fsdp.forward_all(ptrs, pshard.get());
    assert(results.size() == 4);
    std::cout << "[OK] FSDP forward allgathers params" << std::endl;

    fsdp.backward_all(ptrs);
    std::cout << "[OK] FSDP backward reduce-scatters" << std::endl;

    fsdp.step_all(0.01f);
    std::cout << "[OK] FSDP SGD step" << std::endl;
    std::cout << "[OK] FSDP executor tests passed\n" << std::endl;
}

void test_parallelism_factory() {
    std::cout << "=== Testing Parallelism Factory ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i);
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);

    auto dp = ParallelismFactory<float>::create_data_parallel(dist, &coord);
    assert(dp != nullptr); std::cout << "[OK] Data parallel created" << std::endl;

    auto tp = ParallelismFactory<float>::create_tensor_parallel(dist, 4, &coord);
    assert(tp != nullptr); std::cout << "[OK] Tensor parallel created" << std::endl;

    auto pp = ParallelismFactory<float>::create_pipeline_parallel(dist, 4, 2, &coord);
    assert(pp != nullptr); std::cout << "[OK] Pipeline parallel created" << std::endl;

    auto fsdp = ParallelismFactory<float>::create_fsdp(dist, 4, &coord);
    assert(fsdp != nullptr); std::cout << "[OK] FSDP created" << std::endl;

    auto eplb = ParallelismFactory<float>::create_expert_balancer(16, 8, 2, 4, 4, 4);
    assert(eplb != nullptr); std::cout << "[OK] Expert load balancer created" << std::endl;

    auto ep = ParallelismFactory<float>::create_expert_parallel(dist, std::move(eplb));
    assert(ep != nullptr); std::cout << "[OK] Expert parallel created" << std::endl;

    auto p3d = ParallelismFactory<float>::create_3d_parallel(dist, 2, 2, 1, &coord);
    assert(p3d != nullptr); std::cout << "[OK] 3D parallel created" << std::endl;
    std::cout << "[OK] Parallelism factory tests passed\n" << std::endl;
}

void test_moe_expert_assignment() {
    std::cout << "=== Testing MoE Expert Assignment ===" << std::endl;

    ExpertLoadBalancer balancer(256, 32, 4, 8, 32, 16);
    std::vector<double> loads(256);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.5, 5.0);
    for (int i = 0; i < 256; ++i) loads[i] = dis(gen);
    balancer.update_expert_loads(loads);

    auto assignment = balancer.rebalance_experts();
    int total = 0;
    for (int i = 0; i < 256; ++i) { assert(!assignment[i].empty()); total += assignment[i].size(); }
    std::cout << "Total expert assignments: " << total << std::endl;

    std::vector<double> wl(32, 0);
    for (int e = 0; e < 256; ++e)
        for (int w : assignment[e]) wl[w] += loads[e] / assignment[e].size();

    double mx = 0, mn = 1e18;
    for (double l : wl) { mx = std::max(mx, l); mn = std::min(mn, l); }
    std::cout << "Load imbalance: " << (mx / mn) << "x" << std::endl;

    int redundant = 0;
    for (const auto& ws : assignment) if (ws.size() > 1) redundant++;
    std::cout << "Experts with redundancy: " << redundant << std::endl;
    std::cout << "[OK] MoE expert assignment tests passed\n" << std::endl;
}

void test_heartbeat_monitoring() {
    std::cout << "=== Testing Heartbeat Monitoring ===" << std::endl;
    auto lb = std::make_unique<DistributedLoadBalancer<float>>(0.05, 0.2);

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i);
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (auto* w : raw) lb->register_worker(w);

    lb->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    for (int i = 0; i < 4; ++i) {
        auto hb = lb->get_worker_heartbeat(i);
        assert(hb.worker_id == i);
        assert(hb.is_healthy(1.0));
    }
    std::cout << "[OK] Heartbeats tracked" << std::endl;

    workers[2]->mark_failed();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto healthy = lb->get_healthy_workers();
    assert(healthy.size() == 3);
    std::cout << "[OK] Failed worker detected" << std::endl;

    lb->stop();
    std::cout << "[OK] Heartbeat monitoring tests passed\n" << std::endl;
}

void test_expert_parallel_executor() {
    std::cout << "=== Testing Expert Parallel Executor ===" << std::endl;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::expert_parallel(4, i, 8, 2);
        auto w = std::make_unique<DW>(i, cfg);
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);

    auto balancer = ParallelismFactory<float>::create_expert_balancer(8, 4, 1, 4, 2, 0);
    ExpertParallelExecutor<float> ep(dist, std::move(balancer));

    float input[] = {1.0f, 2.0f, 3.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 3);
    auto outputs = ep.route_and_compute(input_tensor.get(), 2);
    assert(!outputs.empty());
    std::cout << "[OK] Expert parallel routing works" << std::endl;

    std::vector<double> new_loads(8, 1.0);
    new_loads[0] = 10.0;
    ep.update_expert_loads(new_loads);
    ep.rebalance();
    std::cout << "[OK] Expert rebalancing works" << std::endl;
    std::cout << "[OK] Expert parallel executor tests passed\n" << std::endl;
}

void test_3d_parallel_executor() {
    std::cout << "=== Testing 3D Parallel Executor ===" << std::endl;

    int tp = 2, dp = 2, pp = 1;
    int ws = tp * dp * pp;

    std::vector<std::unique_ptr<DW>> workers;
    std::vector<DW*> raw;
    for (int i = 0; i < ws; ++i) {
        auto cfg = ParallelConfig::hybrid_3d(tp, dp, pp, i);
        auto w = std::make_unique<DW>(i, cfg);
        float weights[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
        w->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
        raw.push_back(w.get());
        workers.push_back(std::move(w));
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < ws; ++i) workers[i]->set_peers(raw);
    for (int i = 0; i < ws; ++i)
        for (int j = 0; j < ws; ++j)
            if (i != j) workers[i]->set_inbox(j, coord.get_mailbox(i, j));

    Parallel3DExecutor<float> exec(dist, tp, dp, pp, &coord);

    auto tp_g = exec.get_tp_group(0, 0);
    assert(tp_g.size() == static_cast<size_t>(tp));
    std::cout << "[OK] TP group extraction" << std::endl;

    auto dp_g = exec.get_dp_group(0, 0);
    assert(dp_g.size() == static_cast<size_t>(dp));
    std::cout << "[OK] DP group extraction" << std::endl;

    auto pp_g = exec.get_pp_group(0, 0);
    assert(pp_g.size() == static_cast<size_t>(pp));
    std::cout << "[OK] PP group extraction" << std::endl;

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input_tensor = std::make_unique<DenseTensor<float>>(input, 6);
    auto output = exec.forward_3d(input_tensor.get());
    assert(output != nullptr);
    std::cout << "[OK] 3D forward pass" << std::endl;
    std::cout << "[OK] 3D parallel executor tests passed\n" << std::endl;
}

void test_tensor_parallelism_primitives() {
    std::cout << "=== Testing Tensor Parallelism Primitives ===" << std::endl;

    float d0[] = {1.0f, 2.0f}, d1[] = {3.0f, 4.0f};
    float d2[] = {5.0f, 6.0f}, d3[] = {7.0f, 8.0f};
    auto t0 = std::make_unique<DenseTensor<float>>(d0, 2);
    auto t1 = std::make_unique<DenseTensor<float>>(d1, 2);
    auto t2 = std::make_unique<DenseTensor<float>>(d2, 2);
    auto t3 = std::make_unique<DenseTensor<float>>(d3, 2);

    const TensorBase<float>* shards[] = {t0.get(), t1.get(), t2.get(), t3.get()};
    auto* gathered = t0->allgather(shards, 4);
    assert(gathered->total_size() == 8);
    assert(gathered->data_non_volatile()[0] == 1.0f);
    assert(gathered->data_non_volatile()[6] == 7.0f);
    std::cout << "[OK] Allgather works" << std::endl;

    const TensorBase<float>* ms[] = {t0.get(), t1.get(), t2.get(), t3.get()};
    auto* mean_r = t0->allreduce_mean(ms, 4);
    assert(std::abs(mean_r->data_non_volatile()[0] - 4.0f) < 1e-5f);
    assert(std::abs(mean_r->data_non_volatile()[1] - 5.0f) < 1e-5f);
    std::cout << "[OK] Allreduce mean works" << std::endl;

    auto* bc = t3->broadcast(t0.get());
    assert(bc->data_non_volatile()[0] == 1.0f);
    std::cout << "[OK] Broadcast works" << std::endl;

    delete gathered; delete mean_r; delete bc;
    std::cout << "[OK] Tensor parallelism primitive tests passed\n" << std::endl;
}

void test_heterogeneous_backend_workers() {
    std::cout << "=== Testing Heterogeneous Backend Workers ===" << std::endl;

    std::vector<std::unique_ptr<DistributedWorker<float>>> workers;
    std::vector<DistributedWorker<float>*> raw;
    BackendType backends[] = {BackendType::CPU_DENSE, BackendType::CPU_MMAP,
                              BackendType::CPU_DENSE, BackendType::CPU_MMAP};

    for (int i = 0; i < 4; ++i) {
        auto cfg = ParallelConfig::data_parallel(4, i, backends[i]);
        std::unique_ptr<DistributedWorker<float>> w;
        if (backends[i] == BackendType::CPU_MMAP) {
            auto mw = std::make_unique<MW>(i, cfg);
            std::size_t shape[] = {3};
            StreamConfig sc; sc.batch_size = 1024;
            auto model = std::make_unique<MmapTensor<float>>(shape, 1, sc, static_cast<float>(i+1));
            mw->set_model_shard(std::move(model));
            raw.push_back(mw.get());
            workers.push_back(std::move(mw));
        } else {
            auto dw = std::make_unique<DW>(i, cfg);
            float weights[] = {static_cast<float>(i+1), static_cast<float>(i+1), static_cast<float>(i+1)};
            dw->set_model_shard(std::make_unique<DenseTensor<float>>(weights, 3));
            raw.push_back(dw.get());
            workers.push_back(std::move(dw));
        }
    }
    for (int i = 0; i < 4; ++i) {
        auto* dw = dynamic_cast<DW*>(raw[i]);
        if (dw) {
            std::vector<DW*> dw_raw;
            for (auto* r : raw) {
                auto* d = dynamic_cast<DW*>(r);
                if (d) dw_raw.push_back(d);
            }
            dw->set_peers(dw_raw);
        }
    }

    WorkerCoordinator<float> coord;
    std::vector<DistributedWorker<float>*> dist(raw.begin(), raw.end());
    coord.register_workers(dist);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) {
                auto* dw = dynamic_cast<DW*>(raw[i]);
                if (dw) dw->set_inbox(j, coord.get_mailbox(i, j));
            }

    assert(workers[0]->backend_type() == BackendType::CPU_DENSE);
    assert(workers[1]->backend_type() == BackendType::CPU_MMAP);
    std::cout << "[OK] Workers have correct backend types" << std::endl;

    for (int i = 0; i < 4; ++i) {
        auto hb = workers[i]->get_heartbeat();
        assert(hb.backend == backends[i]);
    }
    std::cout << "[OK] Heartbeats report correct backend types" << std::endl;

    DataParallelExecutor<float> executor(dist, &coord);
    std::vector<std::unique_ptr<TensorBase<float>>> dshards;
    std::vector<const TensorBase<float>*> ptrs;
    for (int i = 0; i < 4; ++i) {
        float data[] = {static_cast<float>(i*3+1), static_cast<float>(i*3+2), static_cast<float>(i*3+3)};
        dshards.push_back(std::make_unique<DenseTensor<float>>(data, 3));
        ptrs.push_back(dshards.back().get());
    }

    auto results = executor.forward_all(ptrs);
    assert(results.size() == 4);
    for (int i = 0; i < 4; ++i) {
        float expected = static_cast<float>(i*3+1) * static_cast<float>(i+1);
        assert(std::abs(results[i]->get_element(0) - expected) < 1e-5f);
    }
    std::cout << "[OK] Heterogeneous backend forward pass works" << std::endl;
    std::cout << "[OK] Heterogeneous backend tests passed\n" << std::endl;
}

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  TinyTorch Distributed Parallelism Test Suite" << std::endl;
    std::cout << "================================================\n" << std::endl;

    test_shard_descriptor();
    test_parallel_config();
    test_tensor_mailbox();
    test_dense_tensor_worker();
    test_mmap_tensor_worker();
    test_worker_coordinator();
    test_shard_operations();
    test_tensor_parallelism_primitives();
    test_expert_load_balancer_hierarchical();
    test_expert_load_balancer_global();
    test_distributed_load_balancer();
    test_load_balancing_strategies();
    test_data_parallel_executor();
    test_tensor_parallel_executor();
    test_pipeline_parallel_executor();
    test_fsdp_executor();
    test_parallelism_factory();
    test_moe_expert_assignment();
    test_heartbeat_monitoring();
    test_expert_parallel_executor();
    test_3d_parallel_executor();
    test_heterogeneous_backend_workers();

    std::cout << "================================================" << std::endl;
    std::cout << "  All Parallelism Tests Passed! [OK]" << std::endl;
    std::cout << "================================================\n" << std::endl;

    return 0;
}
