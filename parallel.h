#pragma once
#include "auto_tensor.h"
#include "autograd_tensor.h"
#include "gpu_tensor.h"
#include "cuda_tensor.h"
#include "gpu_kernels.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <future>
#include <random>
#include <map>
#include <set>

// ============================================================================
// TinyTorch Unified Distributed Parallelism Framework
// ============================================================================
// HEARTBEAT SYSTEM (Foundation):
//   Every worker continuously reports real-time capacity metrics:
//     - compute_throughput_ops_per_sec
//     - memory_bandwidth_mb_per_sec
//     - current_queue_depth
//     - effective_capacity_score (normalized 0.0-1.0)
//
// LOAD BALANCER (Built on heartbeats):
//   Uses live heartbeat data to distribute shards to the worker with the
//   highest available capacity at that moment.
//
// EPLB / Other Strategies (Built on load balancer):
//   Expert Parallelism Load Balancer and other strategies use the load
//   balancer's capacity-aware shard assignment as their foundation.
// ============================================================================

// ============================================================================
// Backend Type Enumeration
// ============================================================================

enum class BackendType {
    CPU_DENSE,
    CPU_MMAP,
    CUDA,
    OPENGL,
    OPENCL
};

inline std::string backend_type_to_string(BackendType bt) {
    switch (bt) {
        case BackendType::CPU_DENSE: return "CPU_DENSE";
        case BackendType::CPU_MMAP:  return "CPU_MMAP";
        case BackendType::CUDA:      return "CUDA";
        case BackendType::OPENGL:    return "OPENGL";
        case BackendType::OPENCL:    return "OPENCL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Parallelism Configuration
// ============================================================================

enum class ParallelStrategy {
    DATA_PARALLEL,
    TENSOR_PARALLEL,
    PIPELINE_PARALLEL,
    FSDP,
    EXPERT_PARALLEL,
    HYBRID_3D
};

struct ParallelConfig {
    ParallelStrategy strategy;
    int world_size;
    int rank;
    int tp_size;
    int dp_size;
    int pp_size;
    int ep_size;
    int num_experts;
    int num_redundant_experts;
    int expert_group_size;
    BackendType preferred_backend;

    ParallelConfig()
        : strategy(ParallelStrategy::DATA_PARALLEL),
          world_size(1), rank(0),
          tp_size(1), dp_size(1), pp_size(1), ep_size(1),
          num_experts(0), num_redundant_experts(0), expert_group_size(0),
          preferred_backend(BackendType::CPU_DENSE) {}

    static ParallelConfig data_parallel(int dp_size, int rank, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::DATA_PARALLEL;
        cfg.dp_size = dp_size;
        cfg.world_size = dp_size;
        cfg.rank = rank;
        cfg.preferred_backend = backend;
        return cfg;
    }

    static ParallelConfig tensor_parallel(int tp_size, int rank, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::TENSOR_PARALLEL;
        cfg.tp_size = tp_size;
        cfg.world_size = tp_size;
        cfg.rank = rank;
        cfg.preferred_backend = backend;
        return cfg;
    }

    static ParallelConfig pipeline_parallel(int pp_size, int rank, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::PIPELINE_PARALLEL;
        cfg.pp_size = pp_size;
        cfg.world_size = pp_size;
        cfg.rank = rank;
        cfg.preferred_backend = backend;
        return cfg;
    }

    static ParallelConfig fsdp(int dp_size, int rank, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::FSDP;
        cfg.dp_size = dp_size;
        cfg.world_size = dp_size;
        cfg.rank = rank;
        cfg.preferred_backend = backend;
        return cfg;
    }

    static ParallelConfig expert_parallel(int ep_size, int rank, int num_experts,
                                          int num_redundant = 0, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::EXPERT_PARALLEL;
        cfg.ep_size = ep_size;
        cfg.world_size = ep_size;
        cfg.rank = rank;
        cfg.num_experts = num_experts;
        cfg.num_redundant_experts = num_redundant;
        cfg.preferred_backend = backend;
        return cfg;
    }

    static ParallelConfig hybrid_3d(int tp, int dp, int pp, int rank, BackendType backend = BackendType::CPU_DENSE) {
        ParallelConfig cfg;
        cfg.strategy = ParallelStrategy::HYBRID_3D;
        cfg.tp_size = tp;
        cfg.dp_size = dp;
        cfg.pp_size = pp;
        cfg.world_size = tp * dp * pp;
        cfg.rank = rank;
        cfg.preferred_backend = backend;
        return cfg;
    }

    int tp_rank() const { return rank % tp_size; }
    int dp_rank() const { return (rank / tp_size) % dp_size; }
    int pp_rank() const { return rank / (tp_size * dp_size); }
    int ep_rank() const { return rank % ep_size; }
};

// ============================================================================
// Worker Heartbeat - THE FOUNDATION OF THE LOAD BALANCER
// ============================================================================
// Every worker sends heartbeats containing real-time capacity metrics.
// The load balancer uses these to decide where to send work.

enum class WorkerStatus {
    IDLE,
    COMPUTING,
    SYNCING,
    FAILED,
    OFFLINE
};

struct WorkerHeartbeat {
    int worker_id;
    WorkerStatus status;
    BackendType backend;
    double memory_usage_mb;
    double compute_utilization;
    double last_seen_timestamp;
    int tasks_completed;
    int tasks_failed;
    std::string backend_name;
    std::string device_name;
    std::size_t available_memory_bytes;
    std::size_t total_memory_bytes;

    // === DYNAMIC CAPACITY METRICS (updated each heartbeat) ===
    double compute_throughput_ops_per_sec;   // Operations completed per second since last heartbeat
    double memory_bandwidth_mb_per_sec;      // Memory transfer rate since last heartbeat
    double current_queue_depth;              // Number of pending operations in mailboxes
    double effective_capacity_score;         // Normalized 0.0-1.0: higher = more capacity available

    WorkerHeartbeat()
        : worker_id(-1), status(WorkerStatus::OFFLINE), backend(BackendType::CPU_DENSE),
          memory_usage_mb(0), compute_utilization(0),
          last_seen_timestamp(0), tasks_completed(0), tasks_failed(0),
          available_memory_bytes(0), total_memory_bytes(0),
          compute_throughput_ops_per_sec(0), memory_bandwidth_mb_per_sec(0),
          current_queue_depth(0), effective_capacity_score(0) {}

    bool is_healthy(double timeout_seconds = 30.0) const {
        if (status == WorkerStatus::FAILED || status == WorkerStatus::OFFLINE) return false;
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
        return (now - last_seen_timestamp) < timeout_seconds;
    }
};

// ============================================================================
// Shard Descriptor
// ============================================================================

struct ShardDescriptor {
    std::size_t shard_id;
    std::size_t total_shards;
    std::size_t offset;
    std::size_t count;
    std::vector<std::size_t> shape;
    std::size_t total_size;
    BackendType target_backend;

    ShardDescriptor() : shard_id(0), total_shards(1), offset(0), count(0), total_size(0),
                        target_backend(BackendType::CPU_DENSE) {}

    ShardDescriptor(std::size_t id, std::size_t total, std::size_t off, std::size_t cnt,
                    const std::vector<std::size_t>& s, BackendType backend = BackendType::CPU_DENSE)
        : shard_id(id), total_shards(total), offset(off), count(cnt), shape(s),
          target_backend(backend) {
        total_size = 1;
        for (auto dim : shape) total_size *= dim;
    }

    static ShardDescriptor create_for_rank(std::size_t rank, std::size_t world_size,
                                           const std::vector<std::size_t>& full_shape,
                                           BackendType backend = BackendType::CPU_DENSE) {
        std::size_t total = 1;
        for (auto d : full_shape) total *= d;

        std::size_t shard_size = (total + world_size - 1) / world_size;
        std::size_t offset = rank * shard_size;
        std::size_t count = (offset < total) ? std::min(shard_size, total - offset) : 0;

        return ShardDescriptor(rank, world_size, offset, count, full_shape, backend);
    }
};

// ============================================================================
// Utility: Clone a tensor to any backend
// ============================================================================

template<typename T>
std::unique_ptr<TensorBase<T>> clone_tensor_to_backend(const TensorBase<T>* tensor,
                                                        BackendType target_backend) {
    std::vector<std::size_t> shape;
    auto* s = tensor->shape();
    for (std::size_t i = 0; i < tensor->ndim(); ++i) {
        shape.push_back(s[i]);
    }

    switch (target_backend) {
        case BackendType::CPU_DENSE: {
            auto result = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t i = 0; i < tensor->total_size(); ++i) {
                result->set_element(i, tensor->get_element(i));
            }
            return result;
        }
        case BackendType::CPU_MMAP: {
            StreamConfig cfg;
            cfg.batch_size = std::max((std::size_t)1024, tensor->total_size() / 10);
            auto result = std::make_unique<MmapTensor<T>>(shape.data(), shape.size(), cfg);
            for (std::size_t i = 0; i < tensor->total_size(); ++i) {
                result->set_element(i, tensor->get_element(i));
            }
            return result;
        }
        default:
            auto result = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t i = 0; i < tensor->total_size(); ++i) {
                result->set_element(i, tensor->get_element(i));
            }
            return result;
    }
}

// ============================================================================
// Tensor Mailbox: For inter-worker tensor communication
// ============================================================================

template<typename T>
class TensorMailbox {
private:
    std::queue<std::unique_ptr<TensorBase<T>>> _queue;
    mutable std::mutex _mutex;
    std::condition_variable _cv;

public:
    void send(std::unique_ptr<TensorBase<T>> tensor) {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(tensor));
        _cv.notify_one();
    }

    std::unique_ptr<TensorBase<T>> recv() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return !_queue.empty(); });
        auto tensor = std::move(_queue.front());
        _queue.pop();
        return tensor;
    }

    bool try_recv(std::unique_ptr<TensorBase<T>>& out, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(_mutex);
        if (!_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [this] { return !_queue.empty(); })) {
            return false;
        }
        out = std::move(_queue.front());
        _queue.pop();
        return true;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }
};

// ============================================================================
// Distributed Worker Interface
// ============================================================================

template<typename T>
class DistributedWorker {
public:
    virtual ~DistributedWorker() = default;

    virtual int worker_id() const = 0;
    virtual ParallelConfig config() const = 0;
    virtual BackendType backend_type() const = 0;
    virtual WorkerHeartbeat get_heartbeat() = 0;
    virtual void send_heartbeat() = 0;

    virtual std::unique_ptr<TensorBase<T>> forward_shard(
        const TensorBase<T>* input_shard) = 0;

    virtual std::unique_ptr<TensorBase<T>> backward_shard(
        const TensorBase<T>* grad_output_shard) = 0;

    virtual void allgather(const TensorBase<T>* shard, TensorBase<T>* full) = 0;
    virtual void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) = 0;
    virtual void allreduce_sum(TensorBase<T>* tensor) = 0;
    virtual void allreduce_mean(TensorBase<T>* tensor) = 0;
    virtual void send_tensor(const TensorBase<T>* tensor, int to_rank) = 0;
    virtual std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) = 0;
    virtual void broadcast(TensorBase<T>* tensor, int root_rank) = 0;

    virtual void barrier() = 0;
    virtual void sync_gradients() = 0;
    virtual void sync_parameters() = 0;
    virtual void sgd_step(float lr) = 0;
    virtual void zero_grad() = 0;

    virtual bool is_healthy() const = 0;
    virtual void mark_failed() = 0;
};

// ============================================================================
// Backend-Specific Worker: CPU Dense (with full heartbeat capacity tracking)
// ============================================================================

template<typename T>
class DenseTensorWorker : public DistributedWorker<T> {
private:
    int _worker_id;
    ParallelConfig _config;
    WorkerHeartbeat _heartbeat;
    mutable std::mutex _mutex;
    std::atomic<bool> _healthy;

    std::vector<TensorMailbox<T>*> _inboxes;
    std::vector<DenseTensorWorker<T>*> _peers;

    std::unique_ptr<DenseTensor<T>> _model_shard;
    std::unique_ptr<DenseTensor<T>> _grad_shard;

    // Heartbeat capacity tracking
    std::chrono::steady_clock::time_point _last_heartbeat_time;
    int _tasks_since_last_heartbeat;
    std::size_t _bytes_processed_since_last_heartbeat;

    void record_task(std::size_t bytes) {
        _tasks_since_last_heartbeat++;
        _bytes_processed_since_last_heartbeat += bytes;
    }

    void compute_effective_capacity() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - _last_heartbeat_time).count();
        if (elapsed_sec < 0.001) elapsed_sec = 0.001;

        _heartbeat.compute_throughput_ops_per_sec =
            static_cast<double>(_tasks_since_last_heartbeat) / elapsed_sec;
        _heartbeat.memory_bandwidth_mb_per_sec =
            static_cast<double>(_bytes_processed_since_last_heartbeat) / (1024.0 * 1024.0) / elapsed_sec;

        double total_queue = 0;
        for (auto* inbox : _inboxes) {
            if (inbox) total_queue += static_cast<double>(inbox->size());
        }
        _heartbeat.current_queue_depth = total_queue;

        // Capacity score: weighted combination of metrics
        double throughput_score = std::min(1.0, _heartbeat.compute_throughput_ops_per_sec / 1000.0);
        double memory_score = static_cast<double>(_heartbeat.available_memory_bytes) /
                              std::max(static_cast<std::size_t>(1), _heartbeat.total_memory_bytes);
        double queue_penalty = std::min(1.0, _heartbeat.current_queue_depth / 100.0);
        double utilization_penalty = _heartbeat.compute_utilization;

        // CPU dense baseline capacity
        double backend_baseline = 0.5;

        _heartbeat.effective_capacity_score = backend_baseline *
            (0.3 * throughput_score + 0.3 * memory_score) *
            (1.0 - 0.2 * queue_penalty) *
            (1.0 - 0.2 * utilization_penalty);

        _heartbeat.effective_capacity_score = std::max(0.0, std::min(1.0, _heartbeat.effective_capacity_score));

        _last_heartbeat_time = now;
        _tasks_since_last_heartbeat = 0;
        _bytes_processed_since_last_heartbeat = 0;
    }

public:
    DenseTensorWorker(int worker_id, const ParallelConfig& config)
        : _worker_id(worker_id), _config(config), _healthy(true),
          _tasks_since_last_heartbeat(0), _bytes_processed_since_last_heartbeat(0) {
        _last_heartbeat_time = std::chrono::steady_clock::now();
        _heartbeat.worker_id = worker_id;
        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.backend = BackendType::CPU_DENSE;
        _heartbeat.backend_name = "CPU_DENSE";
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.available_memory_bytes = get_total_system_memory() / std::max(1, config.world_size);
        _heartbeat.total_memory_bytes = get_total_system_memory();
        _heartbeat.device_name = "CPU Dense #" + std::to_string(worker_id);
    }

    void set_peers(const std::vector<DenseTensorWorker<T>*>& peers) {
        _peers = peers;
        _inboxes.resize(peers.size(), nullptr);
    }

    void set_inbox(int from_rank, TensorMailbox<T>* inbox) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size())) {
            _inboxes[from_rank] = inbox;
        }
    }

    void set_model_shard(std::unique_ptr<DenseTensor<T>> shard) {
        _model_shard = std::move(shard);
    }

    int worker_id() const override { return _worker_id; }
    ParallelConfig config() const override { return _config; }
    BackendType backend_type() const override { return BackendType::CPU_DENSE; }
    bool is_healthy() const override { return _healthy.load(); }
    void mark_failed() override {
        _healthy.store(false);
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::FAILED;
    }

    WorkerHeartbeat get_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        compute_effective_capacity();
        return _heartbeat;
    }

    void send_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.status = WorkerStatus::IDLE;
    }

    std::unique_ptr<TensorBase<T>> forward_shard(const TensorBase<T>* input_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::size_t out_size = input_shard->total_size();
        std::vector<std::size_t> out_shape = {out_size};
        auto result = std::make_unique<DenseTensor<T>>(out_shape.data(), out_shape.size());

        if (_model_shard) {
            std::size_t model_size = _model_shard->total_size();
            for (std::size_t i = 0; i < out_size; ++i) {
                T input_val = input_shard->get_element(i);
                T weight_val = _model_shard->get_element(i % model_size);
                result->set_element(i, input_val * weight_val);
            }
        } else {
            for (std::size_t i = 0; i < out_size; ++i) {
                result->set_element(i, input_shard->get_element(i));
            }
        }

        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.tasks_completed++;
        record_task(out_size * sizeof(T));
        return result;
    }

    std::unique_ptr<TensorBase<T>> backward_shard(const TensorBase<T>* grad_output_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::size_t grad_size = grad_output_shard->total_size();
        std::vector<std::size_t> grad_shape = {grad_size};
        auto grad = std::make_unique<DenseTensor<T>>(grad_shape.data(), grad_shape.size());

        for (std::size_t i = 0; i < grad_size; ++i) {
            T grad_out = grad_output_shard->get_element(i);
            T input_val = grad_output_shard->get_element(i);
            grad->set_element(i, grad_out * input_val);
        }

        if (!_grad_shard) {
            _grad_shard = std::make_unique<DenseTensor<T>>(grad_shape.data(), grad_shape.size());
            for (std::size_t i = 0; i < grad_size; ++i) {
                _grad_shard->set_element(i, grad->get_element(i));
            }
        } else {
            for (std::size_t i = 0; i < grad_size; ++i) {
                T existing = _grad_shard->get_element(i);
                T new_val = grad->get_element(i);
                _grad_shard->set_element(i, existing + new_val);
            }
        }

        _heartbeat.status = WorkerStatus::IDLE;
        record_task(grad_size * sizeof(T));
        return grad;
    }

    void allgather(const TensorBase<T>* shard, TensorBase<T>* full) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                full->set_element(offset + i, shard->get_element(i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                shard->set_element(i, full->get_element(offset + i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void allreduce_sum(TensorBase<T>* tensor) override {
        std::size_t size = tensor->total_size();
        std::vector<std::unique_ptr<TensorBase<T>>> received;
        received.reserve(_config.world_size - 1);

        for (int i = 0; i < _config.world_size; ++i) {
            if (i == _worker_id) continue;
            if (i < static_cast<int>(_inboxes.size()) && _inboxes[i]) {
                std::unique_ptr<TensorBase<T>> t;
                if (_inboxes[i]->try_recv(t, 5000)) {
                    received.push_back(std::move(t));
                }
            }
        }

        for (std::size_t i = 0; i < size; ++i) {
            T sum = tensor->get_element(i);
            for (const auto& r : received) {
                if (i < r->total_size()) {
                    sum += r->get_element(i);
                }
            }
            tensor->set_element(i, sum);
        }
        record_task(size * sizeof(T));
    }

    void allreduce_mean(TensorBase<T>* tensor) override {
        allreduce_sum(tensor);
        std::size_t size = tensor->total_size();
        T inv_world_size = T{1} / static_cast<T>(_config.world_size);
        for (std::size_t i = 0; i < size; ++i) {
            tensor->set_element(i, tensor->get_element(i) * inv_world_size);
        }
        record_task(size * sizeof(T));
    }

    void send_tensor(const TensorBase<T>* tensor, int to_rank) override {
        if (to_rank >= 0 && to_rank < static_cast<int>(_peers.size())) {
            std::vector<std::size_t> shape;
            auto* s = tensor->shape();
            for (std::size_t i = 0; i < tensor->ndim(); ++i) shape.push_back(s[i]);
            auto cloned = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t i = 0; i < tensor->total_size(); ++i) {
                cloned->set_element(i, tensor->get_element(i));
            }
            _peers[to_rank]->receive_from(_worker_id, std::move(cloned));
            record_task(tensor->total_size() * sizeof(T));
        }
    }

    std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) override {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            auto t = _inboxes[from_rank]->recv();
            record_task(t->total_size() * sizeof(T));
            return t;
        }
        return nullptr;
    }

    void broadcast(TensorBase<T>* tensor, int root_rank) override {
        if (_worker_id == root_rank) {
            for (int i = 0; i < _config.world_size; ++i) {
                if (i != root_rank) send_tensor(tensor, i);
            }
        } else {
            auto received = recv_tensor(root_rank);
            if (received) {
                for (std::size_t i = 0; i < received->total_size() && i < tensor->total_size(); ++i) {
                    tensor->set_element(i, received->get_element(i));
                }
            }
        }
    }

    void barrier() override {
        static std::mutex barrier_mutex;
        static std::condition_variable barrier_cv;
        static std::atomic<int> barrier_counter{0};
        static std::atomic<int> barrier_generation{0};

        int gen = barrier_generation.load();
        int count = barrier_counter.fetch_add(1) + 1;

        if (count == _config.world_size) {
            barrier_counter.store(0);
            barrier_generation.fetch_add(1);
            barrier_cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            barrier_cv.wait(lock, [gen] { return barrier_generation.load() > gen; });
        }
    }

    void sync_gradients() override {
        if (_grad_shard) {
            allreduce_mean(_grad_shard.get());
        }
    }

    void sync_parameters() override {
        if (_model_shard) {
            broadcast(_model_shard.get(), 0);
        }
    }

    void zero_grad() override {
        if (_grad_shard) {
            std::size_t size = _grad_shard->total_size();
            for (std::size_t i = 0; i < size; ++i) {
                _grad_shard->set_element(i, T{0});
            }
        }
    }

    void sgd_step(float lr) override {
        if (_model_shard && _grad_shard) {
            std::size_t size = std::min(_model_shard->total_size(), _grad_shard->total_size());
            for (std::size_t i = 0; i < size; ++i) {
                T param = _model_shard->get_element(i);
                T grad = _grad_shard->get_element(i);
                _model_shard->set_element(i, param - static_cast<T>(lr) * grad);
            }
        }
    }

    void receive_from(int from_rank, std::unique_ptr<DenseTensor<T>> tensor) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            _inboxes[from_rank]->send(std::move(tensor));
        }
    }

private:
    static double current_timestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
    }
};

// ============================================================================
// Backend-Specific Worker: CPU Mmap (with full heartbeat capacity tracking)
// ============================================================================

template<typename T>
class MmapTensorWorker : public DistributedWorker<T> {
private:
    int _worker_id;
    ParallelConfig _config;
    WorkerHeartbeat _heartbeat;
    mutable std::mutex _mutex;
    std::atomic<bool> _healthy;

    std::vector<TensorMailbox<T>*> _inboxes;
    std::vector<MmapTensorWorker<T>*> _peers;

    std::unique_ptr<MmapTensor<T>> _model_shard;
    std::unique_ptr<MmapTensor<T>> _grad_shard;
    StreamConfig _stream_config;

    // Heartbeat capacity tracking
    std::chrono::steady_clock::time_point _last_heartbeat_time;
    int _tasks_since_last_heartbeat;
    std::size_t _bytes_processed_since_last_heartbeat;

    void record_task(std::size_t bytes) {
        _tasks_since_last_heartbeat++;
        _bytes_processed_since_last_heartbeat += bytes;
    }

    void compute_effective_capacity() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - _last_heartbeat_time).count();
        if (elapsed_sec < 0.001) elapsed_sec = 0.001;

        _heartbeat.compute_throughput_ops_per_sec =
            static_cast<double>(_tasks_since_last_heartbeat) / elapsed_sec;
        _heartbeat.memory_bandwidth_mb_per_sec =
            static_cast<double>(_bytes_processed_since_last_heartbeat) / (1024.0 * 1024.0) / elapsed_sec;

        double total_queue = 0;
        for (auto* inbox : _inboxes) {
            if (inbox) total_queue += static_cast<double>(inbox->size());
        }
        _heartbeat.current_queue_depth = total_queue;

        double throughput_score = std::min(1.0, _heartbeat.compute_throughput_ops_per_sec / 500.0);
        double memory_score = static_cast<double>(_heartbeat.available_memory_bytes) /
                              std::max(static_cast<std::size_t>(1), _heartbeat.total_memory_bytes);
        double queue_penalty = std::min(1.0, _heartbeat.current_queue_depth / 100.0);
        double utilization_penalty = _heartbeat.compute_utilization;

        // Mmap has lower throughput but handles larger tensors
        double backend_baseline = 0.3;

        _heartbeat.effective_capacity_score = backend_baseline *
            (0.3 * throughput_score + 0.3 * memory_score) *
            (1.0 - 0.2 * queue_penalty) *
            (1.0 - 0.2 * utilization_penalty);

        _heartbeat.effective_capacity_score = std::max(0.0, std::min(1.0, _heartbeat.effective_capacity_score));

        _last_heartbeat_time = now;
        _tasks_since_last_heartbeat = 0;
        _bytes_processed_since_last_heartbeat = 0;
    }

public:
    MmapTensorWorker(int worker_id, const ParallelConfig& config)
        : _worker_id(worker_id), _config(config), _healthy(true),
          _tasks_since_last_heartbeat(0), _bytes_processed_since_last_heartbeat(0) {
        _stream_config.batch_size = 1024;
        _last_heartbeat_time = std::chrono::steady_clock::now();
        _heartbeat.worker_id = worker_id;
        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.backend = BackendType::CPU_MMAP;
        _heartbeat.backend_name = "CPU_MMAP";
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.available_memory_bytes = get_total_system_memory() / std::max(1, config.world_size);
        _heartbeat.total_memory_bytes = get_total_system_memory();
        _heartbeat.device_name = "CPU Mmap #" + std::to_string(worker_id);
    }

    void set_peers(const std::vector<MmapTensorWorker<T>*>& peers) {
        _peers = peers;
        _inboxes.resize(peers.size(), nullptr);
    }

    void set_inbox(int from_rank, TensorMailbox<T>* inbox) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size())) {
            _inboxes[from_rank] = inbox;
        }
    }

    void set_model_shard(std::unique_ptr<MmapTensor<T>> shard) {
        _model_shard = std::move(shard);
    }

    int worker_id() const override { return _worker_id; }
    ParallelConfig config() const override { return _config; }
    BackendType backend_type() const override { return BackendType::CPU_MMAP; }
    bool is_healthy() const override { return _healthy.load(); }
    void mark_failed() override {
        _healthy.store(false);
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::FAILED;
    }

    WorkerHeartbeat get_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        compute_effective_capacity();
        return _heartbeat;
    }

    void send_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.status = WorkerStatus::IDLE;
    }

    std::unique_ptr<TensorBase<T>> forward_shard(const TensorBase<T>* input_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::size_t out_size = input_shard->total_size();
        std::vector<std::size_t> out_shape = {out_size};
        auto result = std::make_unique<MmapTensor<T>>(out_shape.data(), out_shape.size(), _stream_config);

        if (_model_shard) {
            std::size_t model_size = _model_shard->total_size();
            for (std::size_t i = 0; i < out_size; ++i) {
                T input_val = input_shard->get_element(i);
                T weight_val = _model_shard->get_element(i % model_size);
                result->set_element(i, input_val * weight_val);
            }
        } else {
            for (std::size_t i = 0; i < out_size; ++i) {
                result->set_element(i, input_shard->get_element(i));
            }
        }

        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.tasks_completed++;
        record_task(out_size * sizeof(T));
        return result;
    }

    std::unique_ptr<TensorBase<T>> backward_shard(const TensorBase<T>* grad_output_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::size_t grad_size = grad_output_shard->total_size();
        std::vector<std::size_t> grad_shape = {grad_size};
        auto grad = std::make_unique<MmapTensor<T>>(grad_shape.data(), grad_shape.size(), _stream_config);

        for (std::size_t i = 0; i < grad_size; ++i) {
            T grad_out = grad_output_shard->get_element(i);
            T input_val = grad_output_shard->get_element(i);
            grad->set_element(i, grad_out * input_val);
        }

        if (!_grad_shard) {
            _grad_shard = std::make_unique<MmapTensor<T>>(grad_shape.data(), grad_shape.size(), _stream_config);
            for (std::size_t i = 0; i < grad_size; ++i) {
                _grad_shard->set_element(i, grad->get_element(i));
            }
        } else {
            for (std::size_t i = 0; i < grad_size; ++i) {
                T existing = _grad_shard->get_element(i);
                T new_val = grad->get_element(i);
                _grad_shard->set_element(i, existing + new_val);
            }
        }

        _heartbeat.status = WorkerStatus::IDLE;
        record_task(grad_size * sizeof(T));
        return grad;
    }

    void allgather(const TensorBase<T>* shard, TensorBase<T>* full) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                full->set_element(offset + i, shard->get_element(i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                shard->set_element(i, full->get_element(offset + i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void allreduce_sum(TensorBase<T>* tensor) override {
        std::size_t size = tensor->total_size();
        std::vector<std::unique_ptr<TensorBase<T>>> received;
        received.reserve(_config.world_size - 1);

        for (int i = 0; i < _config.world_size; ++i) {
            if (i == _worker_id) continue;
            if (i < static_cast<int>(_inboxes.size()) && _inboxes[i]) {
                std::unique_ptr<TensorBase<T>> t;
                if (_inboxes[i]->try_recv(t, 5000)) {
                    received.push_back(std::move(t));
                }
            }
        }

        for (std::size_t i = 0; i < size; ++i) {
            T sum = tensor->get_element(i);
            for (const auto& r : received) {
                if (i < r->total_size()) {
                    sum += r->get_element(i);
                }
            }
            tensor->set_element(i, sum);
        }
        record_task(size * sizeof(T));
    }

    void allreduce_mean(TensorBase<T>* tensor) override {
        allreduce_sum(tensor);
        std::size_t size = tensor->total_size();
        T inv_world_size = T{1} / static_cast<T>(_config.world_size);
        for (std::size_t i = 0; i < size; ++i) {
            tensor->set_element(i, tensor->get_element(i) * inv_world_size);
        }
        record_task(size * sizeof(T));
    }

    void send_tensor(const TensorBase<T>* tensor, int to_rank) override {
        if (to_rank >= 0 && to_rank < static_cast<int>(_peers.size())) {
            std::vector<std::size_t> shape;
            auto* s = tensor->shape();
            for (std::size_t i = 0; i < tensor->ndim(); ++i) shape.push_back(s[i]);
            auto cloned = std::make_unique<MmapTensor<T>>(shape.data(), shape.size(), _stream_config);
            for (std::size_t i = 0; i < tensor->total_size(); ++i) {
                cloned->set_element(i, tensor->get_element(i));
            }
            _peers[to_rank]->receive_from(_worker_id, std::move(cloned));
            record_task(tensor->total_size() * sizeof(T));
        }
    }

    std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) override {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            auto t = _inboxes[from_rank]->recv();
            record_task(t->total_size() * sizeof(T));
            return t;
        }
        return nullptr;
    }

    void broadcast(TensorBase<T>* tensor, int root_rank) override {
        if (_worker_id == root_rank) {
            for (int i = 0; i < _config.world_size; ++i) {
                if (i != root_rank) send_tensor(tensor, i);
            }
        } else {
            auto received = recv_tensor(root_rank);
            if (received) {
                for (std::size_t i = 0; i < received->total_size() && i < tensor->total_size(); ++i) {
                    tensor->set_element(i, received->get_element(i));
                }
            }
        }
    }

    void barrier() override {
        static std::mutex barrier_mutex;
        static std::condition_variable barrier_cv;
        static std::atomic<int> barrier_counter{0};
        static std::atomic<int> barrier_generation{0};

        int gen = barrier_generation.load();
        int count = barrier_counter.fetch_add(1) + 1;

        if (count == _config.world_size) {
            barrier_counter.store(0);
            barrier_generation.fetch_add(1);
            barrier_cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            barrier_cv.wait(lock, [gen] { return barrier_generation.load() > gen; });
        }
    }

    void sync_gradients() override {
        if (_grad_shard) {
            allreduce_mean(_grad_shard.get());
        }
    }

    void sync_parameters() override {
        if (_model_shard) {
            broadcast(_model_shard.get(), 0);
        }
    }

    void zero_grad() override {
        if (_grad_shard) {
            std::size_t size = _grad_shard->total_size();
            for (std::size_t i = 0; i < size; ++i) {
                _grad_shard->set_element(i, T{0});
            }
        }
    }

    void sgd_step(float lr) override {
        if (_model_shard && _grad_shard) {
            std::size_t size = std::min(_model_shard->total_size(), _grad_shard->total_size());
            for (std::size_t i = 0; i < size; ++i) {
                T param = _model_shard->get_element(i);
                T grad = _grad_shard->get_element(i);
                _model_shard->set_element(i, param - static_cast<T>(lr) * grad);
            }
        }
    }

    void receive_from(int from_rank, std::unique_ptr<MmapTensor<T>> tensor) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            _inboxes[from_rank]->send(std::move(tensor));
        }
    }

private:
    static double current_timestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
    }
};

// ============================================================================
// Backend-Specific Worker: CUDA (with full heartbeat capacity tracking)
// ============================================================================

template<typename T>
class CudaTensorWorker : public DistributedWorker<T> {
private:
    int _worker_id;
    ParallelConfig _config;
    WorkerHeartbeat _heartbeat;
    mutable std::mutex _mutex;
    std::atomic<bool> _healthy;
    int _device_id;

    std::vector<TensorMailbox<T>*> _inboxes;
    std::vector<CudaTensorWorker<T>*> _peers;

    std::unique_ptr<CudaTensor> _model_shard;
    std::unique_ptr<CudaTensor> _grad_shard;

    // Heartbeat capacity tracking
    std::chrono::steady_clock::time_point _last_heartbeat_time;
    int _tasks_since_last_heartbeat;
    std::size_t _bytes_processed_since_last_heartbeat;

    void record_task(std::size_t bytes) {
        _tasks_since_last_heartbeat++;
        _bytes_processed_since_last_heartbeat += bytes;
    }

    void compute_effective_capacity() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - _last_heartbeat_time).count();
        if (elapsed_sec < 0.001) elapsed_sec = 0.001;

        _heartbeat.compute_throughput_ops_per_sec =
            static_cast<double>(_tasks_since_last_heartbeat) / elapsed_sec;
        _heartbeat.memory_bandwidth_mb_per_sec =
            static_cast<double>(_bytes_processed_since_last_heartbeat) / (1024.0 * 1024.0) / elapsed_sec;

        double total_queue = 0;
        for (auto* inbox : _inboxes) {
            if (inbox) total_queue += static_cast<double>(inbox->size());
        }
        _heartbeat.current_queue_depth = total_queue;

        double throughput_score = std::min(1.0, _heartbeat.compute_throughput_ops_per_sec / 5000.0);
        double memory_score = static_cast<double>(_heartbeat.available_memory_bytes) /
                              std::max(static_cast<std::size_t>(1), _heartbeat.total_memory_bytes);
        double queue_penalty = std::min(1.0, _heartbeat.current_queue_depth / 100.0);
        double utilization_penalty = _heartbeat.compute_utilization;

        // CUDA has high throughput but limited by PCIe bandwidth
        double backend_baseline = 0.8;

        _heartbeat.effective_capacity_score = backend_baseline *
            (0.4 * throughput_score + 0.2 * memory_score) *
            (1.0 - 0.2 * queue_penalty) *
            (1.0 - 0.2 * utilization_penalty);

        _heartbeat.effective_capacity_score = std::max(0.0, std::min(1.0, _heartbeat.effective_capacity_score));

        _last_heartbeat_time = now;
        _tasks_since_last_heartbeat = 0;
        _bytes_processed_since_last_heartbeat = 0;
    }

public:
    CudaTensorWorker(int worker_id, const ParallelConfig& config, int device_id = 0)
        : _worker_id(worker_id), _config(config), _healthy(true), _device_id(device_id),
          _tasks_since_last_heartbeat(0), _bytes_processed_since_last_heartbeat(0) {
        _last_heartbeat_time = std::chrono::steady_clock::now();
        _heartbeat.worker_id = worker_id;
        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.backend = BackendType::CUDA;
        _heartbeat.backend_name = "CUDA";
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.available_memory_bytes = get_total_system_memory() / std::max(1, config.world_size);
        _heartbeat.total_memory_bytes = get_total_system_memory();
        _heartbeat.device_name = "CUDA Device #" + std::to_string(device_id);
    }

    void set_peers(const std::vector<CudaTensorWorker<T>*>& peers) {
        _peers = peers;
        _inboxes.resize(peers.size(), nullptr);
    }

    void set_inbox(int from_rank, TensorMailbox<T>* inbox) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size())) {
            _inboxes[from_rank] = inbox;
        }
    }

    void set_model_shard(std::unique_ptr<CudaTensor> shard) {
        _model_shard = std::move(shard);
    }

    int worker_id() const override { return _worker_id; }
    ParallelConfig config() const override { return _config; }
    BackendType backend_type() const override { return BackendType::CUDA; }
    bool is_healthy() const override { return _healthy.load(); }
    void mark_failed() override {
        _healthy.store(false);
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::FAILED;
    }

    WorkerHeartbeat get_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        compute_effective_capacity();
        return _heartbeat;
    }

    void send_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.status = WorkerStatus::IDLE;
    }

    std::unique_ptr<TensorBase<T>> forward_shard(const TensorBase<T>* input_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> result;

        if (_model_shard) {
            result = _model_shard->multiply(input_shard);
        } else {
            result = clone_tensor_to_backend(input_shard, BackendType::CPU_DENSE);
        }

        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.tasks_completed++;
        record_task(input_shard->total_size() * sizeof(T));
        return result;
    }

    std::unique_ptr<TensorBase<T>> backward_shard(const TensorBase<T>* grad_output_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> grad;

        if (_model_shard) {
            grad = _model_shard->multiply(grad_output_shard);
        } else {
            grad = clone_tensor_to_backend(grad_output_shard, BackendType::CPU_DENSE);
        }

        if (!_grad_shard) {
            _grad_shard = std::unique_ptr<CudaTensor>(static_cast<CudaTensor*>(grad.release()));
        } else {
            auto sum_result = _grad_shard->add(grad.get());
            _grad_shard = std::unique_ptr<CudaTensor>(static_cast<CudaTensor*>(sum_result.release()));
        }

        _heartbeat.status = WorkerStatus::IDLE;
        record_task(grad_output_shard->total_size() * sizeof(T));
        return grad;
    }

    void allgather(const TensorBase<T>* shard, TensorBase<T>* full) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                full->set_element(offset + i, shard->get_element(i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                shard->set_element(i, full->get_element(offset + i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void allreduce_sum(TensorBase<T>* tensor) override {
        std::size_t size = tensor->total_size();
        std::vector<std::unique_ptr<TensorBase<T>>> received;
        received.reserve(_config.world_size - 1);

        for (int i = 0; i < _config.world_size; ++i) {
            if (i == _worker_id) continue;
            if (i < static_cast<int>(_inboxes.size()) && _inboxes[i]) {
                std::unique_ptr<TensorBase<T>> t;
                if (_inboxes[i]->try_recv(t, 5000)) {
                    received.push_back(std::move(t));
                }
            }
        }

        for (std::size_t i = 0; i < size; ++i) {
            T sum = tensor->get_element(i);
            for (const auto& r : received) {
                if (i < r->total_size()) {
                    sum += r->get_element(i);
                }
            }
            tensor->set_element(i, sum);
        }
        record_task(size * sizeof(T));
    }

    void allreduce_mean(TensorBase<T>* tensor) override {
        allreduce_sum(tensor);
        std::size_t size = tensor->total_size();
        T inv_world_size = T{1} / static_cast<T>(_config.world_size);
        for (std::size_t i = 0; i < size; ++i) {
            tensor->set_element(i, tensor->get_element(i) * inv_world_size);
        }
        record_task(size * sizeof(T));
    }

    void send_tensor(const TensorBase<T>* tensor, int to_rank) override {
        if (to_rank >= 0 && to_rank < static_cast<int>(_peers.size())) {
            auto cloned = clone_tensor_to_backend(tensor, BackendType::CPU_DENSE);
            _peers[to_rank]->receive_from(_worker_id, std::move(cloned));
            record_task(tensor->total_size() * sizeof(T));
        }
    }

    std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) override {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            auto t = _inboxes[from_rank]->recv();
            record_task(t->total_size() * sizeof(T));
            return t;
        }
        return nullptr;
    }

    void broadcast(TensorBase<T>* tensor, int root_rank) override {
        if (_worker_id == root_rank) {
            for (int i = 0; i < _config.world_size; ++i) {
                if (i != root_rank) send_tensor(tensor, i);
            }
        } else {
            auto received = recv_tensor(root_rank);
            if (received) {
                for (std::size_t i = 0; i < received->total_size() && i < tensor->total_size(); ++i) {
                    tensor->set_element(i, received->get_element(i));
                }
            }
        }
    }

    void barrier() override {
        static std::mutex barrier_mutex;
        static std::condition_variable barrier_cv;
        static std::atomic<int> barrier_counter{0};
        static std::atomic<int> barrier_generation{0};

        int gen = barrier_generation.load();
        int count = barrier_counter.fetch_add(1) + 1;

        if (count == _config.world_size) {
            barrier_counter.store(0);
            barrier_generation.fetch_add(1);
            barrier_cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            barrier_cv.wait(lock, [gen] { return barrier_generation.load() > gen; });
        }
    }

    void sync_gradients() override {
        if (_grad_shard) {
            allreduce_mean(_grad_shard.get());
        }
    }

    void sync_parameters() override {
        if (_model_shard) {
            broadcast(_model_shard.get(), 0);
        }
    }

    void zero_grad() override {
        if (_grad_shard) {
            std::size_t size = _grad_shard->total_size();
            for (std::size_t i = 0; i < size; ++i) {
                _grad_shard->set_element(i, T{0});
            }
        }
    }

    void sgd_step(float lr) override {
        if (_model_shard && _grad_shard) {
            std::size_t size = std::min(_model_shard->total_size(), _grad_shard->total_size());
            for (std::size_t i = 0; i < size; ++i) {
                T param = _model_shard->get_element(i);
                T grad = _grad_shard->get_element(i);
                _model_shard->set_element(i, param - static_cast<T>(lr) * grad);
            }
        }
    }

    void receive_from(int from_rank, std::unique_ptr<TensorBase<T>> tensor) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            _inboxes[from_rank]->send(std::move(tensor));
        }
    }

private:
    static double current_timestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
    }
};

// ============================================================================
// Backend-Specific Worker: OpenGL (with full heartbeat capacity tracking)
// ============================================================================

template<typename T>
class GpuTensorWorker : public DistributedWorker<T> {
private:
    int _worker_id;
    ParallelConfig _config;
    WorkerHeartbeat _heartbeat;
    mutable std::mutex _mutex;
    std::atomic<bool> _healthy;
    int _device_id;

    std::vector<TensorMailbox<T>*> _inboxes;
    std::vector<GpuTensorWorker<T>*> _peers;

    std::unique_ptr<GpuTensor<T>> _model_shard;
    std::unique_ptr<GpuTensor<T>> _grad_shard;

    // Heartbeat capacity tracking
    std::chrono::steady_clock::time_point _last_heartbeat_time;
    int _tasks_since_last_heartbeat;
    std::size_t _bytes_processed_since_last_heartbeat;

    void record_task(std::size_t bytes) {
        _tasks_since_last_heartbeat++;
        _bytes_processed_since_last_heartbeat += bytes;
    }

    void compute_effective_capacity() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - _last_heartbeat_time).count();
        if (elapsed_sec < 0.001) elapsed_sec = 0.001;

        _heartbeat.compute_throughput_ops_per_sec =
            static_cast<double>(_tasks_since_last_heartbeat) / elapsed_sec;
        _heartbeat.memory_bandwidth_mb_per_sec =
            static_cast<double>(_bytes_processed_since_last_heartbeat) / (1024.0 * 1024.0) / elapsed_sec;

        double total_queue = 0;
        for (auto* inbox : _inboxes) {
            if (inbox) total_queue += static_cast<double>(inbox->size());
        }
        _heartbeat.current_queue_depth = total_queue;

        double throughput_score = std::min(1.0, _heartbeat.compute_throughput_ops_per_sec / 3000.0);
        double memory_score = static_cast<double>(_heartbeat.available_memory_bytes) /
                              std::max(static_cast<std::size_t>(1), _heartbeat.total_memory_bytes);
        double queue_penalty = std::min(1.0, _heartbeat.current_queue_depth / 100.0);
        double utilization_penalty = _heartbeat.compute_utilization;

        // OpenGL has moderate throughput
        double backend_baseline = 0.6;

        _heartbeat.effective_capacity_score = backend_baseline *
            (0.35 * throughput_score + 0.25 * memory_score) *
            (1.0 - 0.2 * queue_penalty) *
            (1.0 - 0.2 * utilization_penalty);

        _heartbeat.effective_capacity_score = std::max(0.0, std::min(1.0, _heartbeat.effective_capacity_score));

        _last_heartbeat_time = now;
        _tasks_since_last_heartbeat = 0;
        _bytes_processed_since_last_heartbeat = 0;
    }

public:
    GpuTensorWorker(int worker_id, const ParallelConfig& config, int device_id = 0)
        : _worker_id(worker_id), _config(config), _healthy(true), _device_id(device_id),
          _tasks_since_last_heartbeat(0), _bytes_processed_since_last_heartbeat(0) {
        _last_heartbeat_time = std::chrono::steady_clock::now();
        _heartbeat.worker_id = worker_id;
        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.backend = BackendType::OPENGL;
        _heartbeat.backend_name = "OPENGL";
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.available_memory_bytes = get_total_system_memory() / std::max(1, config.world_size);
        _heartbeat.total_memory_bytes = get_total_system_memory();
        _heartbeat.device_name = "OpenGL GPU #" + std::to_string(device_id);
    }

    void set_peers(const std::vector<GpuTensorWorker<T>*>& peers) {
        _peers = peers;
        _inboxes.resize(peers.size(), nullptr);
    }

    void set_inbox(int from_rank, TensorMailbox<T>* inbox) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size())) {
            _inboxes[from_rank] = inbox;
        }
    }

    void set_model_shard(std::unique_ptr<GpuTensor<T>> shard) {
        _model_shard = std::move(shard);
    }

    int worker_id() const override { return _worker_id; }
    ParallelConfig config() const override { return _config; }
    BackendType backend_type() const override { return BackendType::OPENGL; }
    bool is_healthy() const override { return _healthy.load(); }
    void mark_failed() override {
        _healthy.store(false);
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::FAILED;
    }

    WorkerHeartbeat get_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        compute_effective_capacity();
        return _heartbeat;
    }

    void send_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.status = WorkerStatus::IDLE;
    }

    std::unique_ptr<TensorBase<T>> forward_shard(const TensorBase<T>* input_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> result;

        if (_model_shard) {
            result = _model_shard->multiply(input_shard);
        } else {
            result = clone_tensor_to_backend(input_shard, BackendType::CPU_DENSE);
        }

        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.tasks_completed++;
        record_task(input_shard->total_size() * sizeof(T));
        return result;
    }

    std::unique_ptr<TensorBase<T>> backward_shard(const TensorBase<T>* grad_output_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> grad;

        if (_model_shard) {
            grad = _model_shard->multiply(grad_output_shard);
        } else {
            grad = clone_tensor_to_backend(grad_output_shard, BackendType::CPU_DENSE);
        }

        if (!_grad_shard) {
            _grad_shard = std::unique_ptr<GpuTensor<T>>(static_cast<GpuTensor<T>*>(grad.release()));
        } else {
            auto sum_result = _grad_shard->add(grad.get());
            _grad_shard = std::unique_ptr<GpuTensor<T>>(static_cast<GpuTensor<T>*>(sum_result.release()));
        }

        _heartbeat.status = WorkerStatus::IDLE;
        record_task(grad_output_shard->total_size() * sizeof(T));
        return grad;
    }

    void allgather(const TensorBase<T>* shard, TensorBase<T>* full) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                full->set_element(offset + i, shard->get_element(i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                shard->set_element(i, full->get_element(offset + i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void allreduce_sum(TensorBase<T>* tensor) override {
        std::size_t size = tensor->total_size();
        std::vector<std::unique_ptr<TensorBase<T>>> received;
        received.reserve(_config.world_size - 1);

        for (int i = 0; i < _config.world_size; ++i) {
            if (i == _worker_id) continue;
            if (i < static_cast<int>(_inboxes.size()) && _inboxes[i]) {
                std::unique_ptr<TensorBase<T>> t;
                if (_inboxes[i]->try_recv(t, 5000)) {
                    received.push_back(std::move(t));
                }
            }
        }

        for (std::size_t i = 0; i < size; ++i) {
            T sum = tensor->get_element(i);
            for (const auto& r : received) {
                if (i < r->total_size()) {
                    sum += r->get_element(i);
                }
            }
            tensor->set_element(i, sum);
        }
        record_task(size * sizeof(T));
    }

    void allreduce_mean(TensorBase<T>* tensor) override {
        allreduce_sum(tensor);
        std::size_t size = tensor->total_size();
        T inv_world_size = T{1} / static_cast<T>(_config.world_size);
        for (std::size_t i = 0; i < size; ++i) {
            tensor->set_element(i, tensor->get_element(i) * inv_world_size);
        }
        record_task(size * sizeof(T));
    }

    void send_tensor(const TensorBase<T>* tensor, int to_rank) override {
        if (to_rank >= 0 && to_rank < static_cast<int>(_peers.size())) {
            auto cloned = clone_tensor_to_backend(tensor, BackendType::CPU_DENSE);
            _peers[to_rank]->receive_from(_worker_id, std::move(cloned));
            record_task(tensor->total_size() * sizeof(T));
        }
    }

    std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) override {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            auto t = _inboxes[from_rank]->recv();
            record_task(t->total_size() * sizeof(T));
            return t;
        }
        return nullptr;
    }

    void broadcast(TensorBase<T>* tensor, int root_rank) override {
        if (_worker_id == root_rank) {
            for (int i = 0; i < _config.world_size; ++i) {
                if (i != root_rank) send_tensor(tensor, i);
            }
        } else {
            auto received = recv_tensor(root_rank);
            if (received) {
                for (std::size_t i = 0; i < received->total_size() && i < tensor->total_size(); ++i) {
                    tensor->set_element(i, received->get_element(i));
                }
            }
        }
    }

    void barrier() override {
        static std::mutex barrier_mutex;
        static std::condition_variable barrier_cv;
        static std::atomic<int> barrier_counter{0};
        static std::atomic<int> barrier_generation{0};

        int gen = barrier_generation.load();
        int count = barrier_counter.fetch_add(1) + 1;

        if (count == _config.world_size) {
            barrier_counter.store(0);
            barrier_generation.fetch_add(1);
            barrier_cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            barrier_cv.wait(lock, [gen] { return barrier_generation.load() > gen; });
        }
    }

    void sync_gradients() override {
        if (_grad_shard) {
            allreduce_mean(_grad_shard.get());
        }
    }

    void sync_parameters() override {
        if (_model_shard) {
            broadcast(_model_shard.get(), 0);
        }
    }

    void zero_grad() override {
        if (_grad_shard) {
            std::size_t size = _grad_shard->total_size();
            for (std::size_t i = 0; i < size; ++i) {
                _grad_shard->set_element(i, T{0});
            }
        }
    }

    void sgd_step(float lr) override {
        if (_model_shard && _grad_shard) {
            std::size_t size = std::min(_model_shard->total_size(), _grad_shard->total_size());
            for (std::size_t i = 0; i < size; ++i) {
                T param = _model_shard->get_element(i);
                T grad = _grad_shard->get_element(i);
                _model_shard->set_element(i, param - static_cast<T>(lr) * grad);
            }
        }
    }

    void receive_from(int from_rank, std::unique_ptr<TensorBase<T>> tensor) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            _inboxes[from_rank]->send(std::move(tensor));
        }
    }

private:
    static double current_timestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
    }
};

// ============================================================================
// Backend-Specific Worker: OpenCL (with full heartbeat capacity tracking)
// ============================================================================

template<typename T>
class OpenClTensorWorker : public DistributedWorker<T> {
private:
    int _worker_id;
    ParallelConfig _config;
    WorkerHeartbeat _heartbeat;
    mutable std::mutex _mutex;
    std::atomic<bool> _healthy;
    int _device_id;

    std::vector<TensorMailbox<T>*> _inboxes;
    std::vector<OpenClTensorWorker<T>*> _peers;

    std::unique_ptr<OpenClTensor> _model_shard;
    std::unique_ptr<OpenClTensor> _grad_shard;

    // Heartbeat capacity tracking
    std::chrono::steady_clock::time_point _last_heartbeat_time;
    int _tasks_since_last_heartbeat;
    std::size_t _bytes_processed_since_last_heartbeat;

    void record_task(std::size_t bytes) {
        _tasks_since_last_heartbeat++;
        _bytes_processed_since_last_heartbeat += bytes;
    }

    void compute_effective_capacity() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - _last_heartbeat_time).count();
        if (elapsed_sec < 0.001) elapsed_sec = 0.001;

        _heartbeat.compute_throughput_ops_per_sec =
            static_cast<double>(_tasks_since_last_heartbeat) / elapsed_sec;
        _heartbeat.memory_bandwidth_mb_per_sec =
            static_cast<double>(_bytes_processed_since_last_heartbeat) / (1024.0 * 1024.0) / elapsed_sec;

        double total_queue = 0;
        for (auto* inbox : _inboxes) {
            if (inbox) total_queue += static_cast<double>(inbox->size());
        }
        _heartbeat.current_queue_depth = total_queue;

        double throughput_score = std::min(1.0, _heartbeat.compute_throughput_ops_per_sec / 4000.0);
        double memory_score = static_cast<double>(_heartbeat.available_memory_bytes) /
                              std::max(static_cast<std::size_t>(1), _heartbeat.total_memory_bytes);
        double queue_penalty = std::min(1.0, _heartbeat.current_queue_depth / 100.0);
        double utilization_penalty = _heartbeat.compute_utilization;

        // OpenCL has good cross-platform support
        double backend_baseline = 0.7;

        _heartbeat.effective_capacity_score = backend_baseline *
            (0.35 * throughput_score + 0.25 * memory_score) *
            (1.0 - 0.2 * queue_penalty) *
            (1.0 - 0.2 * utilization_penalty);

        _heartbeat.effective_capacity_score = std::max(0.0, std::min(1.0, _heartbeat.effective_capacity_score));

        _last_heartbeat_time = now;
        _tasks_since_last_heartbeat = 0;
        _bytes_processed_since_last_heartbeat = 0;
    }

public:
    OpenClTensorWorker(int worker_id, const ParallelConfig& config, int device_id = 0)
        : _worker_id(worker_id), _config(config), _healthy(true), _device_id(device_id),
          _tasks_since_last_heartbeat(0), _bytes_processed_since_last_heartbeat(0) {
        _last_heartbeat_time = std::chrono::steady_clock::now();
        _heartbeat.worker_id = worker_id;
        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.backend = BackendType::OPENCL;
        _heartbeat.backend_name = "OPENCL";
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.available_memory_bytes = get_total_system_memory() / std::max(1, config.world_size);
        _heartbeat.total_memory_bytes = get_total_system_memory();
        _heartbeat.device_name = "OpenCL Device #" + std::to_string(device_id);
    }

    void set_peers(const std::vector<OpenClTensorWorker<T>*>& peers) {
        _peers = peers;
        _inboxes.resize(peers.size(), nullptr);
    }

    void set_inbox(int from_rank, TensorMailbox<T>* inbox) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size())) {
            _inboxes[from_rank] = inbox;
        }
    }

    void set_model_shard(std::unique_ptr<OpenClTensor> shard) {
        _model_shard = std::move(shard);
    }

    int worker_id() const override { return _worker_id; }
    ParallelConfig config() const override { return _config; }
    BackendType backend_type() const override { return BackendType::OPENCL; }
    bool is_healthy() const override { return _healthy.load(); }
    void mark_failed() override {
        _healthy.store(false);
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::FAILED;
    }

    WorkerHeartbeat get_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        compute_effective_capacity();
        return _heartbeat;
    }

    void send_heartbeat() override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.last_seen_timestamp = current_timestamp();
        _heartbeat.status = WorkerStatus::IDLE;
    }

    std::unique_ptr<TensorBase<T>> forward_shard(const TensorBase<T>* input_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> result;

        if (_model_shard) {
            result = _model_shard->multiply(input_shard);
        } else {
            result = clone_tensor_to_backend(input_shard, BackendType::CPU_DENSE);
        }

        _heartbeat.status = WorkerStatus::IDLE;
        _heartbeat.tasks_completed++;
        record_task(input_shard->total_size() * sizeof(T));
        return result;
    }

    std::unique_ptr<TensorBase<T>> backward_shard(const TensorBase<T>* grad_output_shard) override {
        std::lock_guard<std::mutex> lock(_mutex);
        _heartbeat.status = WorkerStatus::COMPUTING;

        std::unique_ptr<TensorBase<T>> grad;

        if (_model_shard) {
            grad = _model_shard->multiply(grad_output_shard);
        } else {
            grad = clone_tensor_to_backend(grad_output_shard, BackendType::CPU_DENSE);
        }

        if (!_grad_shard) {
            _grad_shard = std::unique_ptr<OpenClTensor>(static_cast<OpenClTensor*>(grad.release()));
        } else {
            auto sum_result = _grad_shard->add(grad.get());
            _grad_shard = std::unique_ptr<OpenClTensor>(static_cast<OpenClTensor*>(sum_result.release()));
        }

        _heartbeat.status = WorkerStatus::IDLE;
        record_task(grad_output_shard->total_size() * sizeof(T));
        return grad;
    }

    void allgather(const TensorBase<T>* shard, TensorBase<T>* full) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                full->set_element(offset + i, shard->get_element(i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void reducescatter(const TensorBase<T>* full, TensorBase<T>* shard) override {
        std::size_t shard_size = shard->total_size();
        std::size_t offset = _worker_id * shard_size;
        for (std::size_t i = 0; i < shard_size; ++i) {
            if (offset + i < full->total_size()) {
                shard->set_element(i, full->get_element(offset + i));
            }
        }
        record_task(shard_size * sizeof(T));
    }

    void allreduce_sum(TensorBase<T>* tensor) override {
        std::size_t size = tensor->total_size();
        std::vector<std::unique_ptr<TensorBase<T>>> received;
        received.reserve(_config.world_size - 1);

        for (int i = 0; i < _config.world_size; ++i) {
            if (i == _worker_id) continue;
            if (i < static_cast<int>(_inboxes.size()) && _inboxes[i]) {
                std::unique_ptr<TensorBase<T>> t;
                if (_inboxes[i]->try_recv(t, 5000)) {
                    received.push_back(std::move(t));
                }
            }
        }

        for (std::size_t i = 0; i < size; ++i) {
            T sum = tensor->get_element(i);
            for (const auto& r : received) {
                if (i < r->total_size()) {
                    sum += r->get_element(i);
                }
            }
            tensor->set_element(i, sum);
        }
        record_task(size * sizeof(T));
    }

    void allreduce_mean(TensorBase<T>* tensor) override {
        allreduce_sum(tensor);
        std::size_t size = tensor->total_size();
        T inv_world_size = T{1} / static_cast<T>(_config.world_size);
        for (std::size_t i = 0; i < size; ++i) {
            tensor->set_element(i, tensor->get_element(i) * inv_world_size);
        }
        record_task(size * sizeof(T));
    }

    void send_tensor(const TensorBase<T>* tensor, int to_rank) override {
        if (to_rank >= 0 && to_rank < static_cast<int>(_peers.size())) {
            auto cloned = clone_tensor_to_backend(tensor, BackendType::CPU_DENSE);
            _peers[to_rank]->receive_from(_worker_id, std::move(cloned));
            record_task(tensor->total_size() * sizeof(T));
        }
    }

    std::unique_ptr<TensorBase<T>> recv_tensor(int from_rank) override {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            auto t = _inboxes[from_rank]->recv();
            record_task(t->total_size() * sizeof(T));
            return t;
        }
        return nullptr;
    }

    void broadcast(TensorBase<T>* tensor, int root_rank) override {
        if (_worker_id == root_rank) {
            for (int i = 0; i < _config.world_size; ++i) {
                if (i != root_rank) send_tensor(tensor, i);
            }
        } else {
            auto received = recv_tensor(root_rank);
            if (received) {
                for (std::size_t i = 0; i < received->total_size() && i < tensor->total_size(); ++i) {
                    tensor->set_element(i, received->get_element(i));
                }
            }
        }
    }

    void barrier() override {
        static std::mutex barrier_mutex;
        static std::condition_variable barrier_cv;
        static std::atomic<int> barrier_counter{0};
        static std::atomic<int> barrier_generation{0};

        int gen = barrier_generation.load();
        int count = barrier_counter.fetch_add(1) + 1;

        if (count == _config.world_size) {
            barrier_counter.store(0);
            barrier_generation.fetch_add(1);
            barrier_cv.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(barrier_mutex);
            barrier_cv.wait(lock, [gen] { return barrier_generation.load() > gen; });
        }
    }

    void sync_gradients() override {
        if (_grad_shard) {
            allreduce_mean(_grad_shard.get());
        }
    }

    void sync_parameters() override {
        if (_model_shard) {
            broadcast(_model_shard.get(), 0);
        }
    }

    void zero_grad() override {
        if (_grad_shard) {
            std::size_t size = _grad_shard->total_size();
            for (std::size_t i = 0; i < size; ++i) {
                _grad_shard->set_element(i, T{0});
            }
        }
    }

    void sgd_step(float lr) override {
        if (_model_shard && _grad_shard) {
            std::size_t size = std::min(_model_shard->total_size(), _grad_shard->total_size());
            for (std::size_t i = 0; i < size; ++i) {
                T param = _model_shard->get_element(i);
                T grad = _grad_shard->get_element(i);
                _model_shard->set_element(i, param - static_cast<T>(lr) * grad);
            }
        }
    }

    void receive_from(int from_rank, std::unique_ptr<TensorBase<T>> tensor) {
        if (from_rank >= 0 && from_rank < static_cast<int>(_inboxes.size()) && _inboxes[from_rank]) {
            _inboxes[from_rank]->send(std::move(tensor));
        }
    }

private:
    static double current_timestamp() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
    }
};

// ============================================================================
// Expert Load Info
// ============================================================================

struct ExpertLoadInfo {
    int expert_id;
    int group_id;
    double estimated_load;
    int replication_count;
    std::vector<int> assigned_workers;

    ExpertLoadInfo() : expert_id(-1), group_id(0), estimated_load(1.0), replication_count(1) {}
};

// ============================================================================
// Expert Load Balancer (EPLB) - DeepSeek-V3 Style
// ============================================================================
// BUILDS ON TOP OF THE HEARTBEAT SYSTEM.
// Uses worker heartbeats to determine expert placement.

class ExpertLoadBalancer {
private:
    int _num_experts;
    int _num_workers;
    int _num_nodes;
    int _workers_per_node;
    int _num_redundant;
    int _expert_group_size;

    std::vector<ExpertLoadInfo> _expert_loads;
    std::vector<WorkerHeartbeat> _worker_heartbeats;
    mutable std::mutex _mutex;

public:
    ExpertLoadBalancer(int num_experts, int num_workers, int num_nodes,
                       int workers_per_node, int num_redundant = 0,
                       int expert_group_size = 0)
        : _num_experts(num_experts), _num_workers(num_workers),
          _num_nodes(num_nodes), _workers_per_node(workers_per_node),
          _num_redundant(num_redundant), _expert_group_size(expert_group_size) {

        _expert_loads.resize(num_experts);
        _worker_heartbeats.resize(num_workers);

        for (int i = 0; i < num_experts; ++i) {
            _expert_loads[i].expert_id = i;
            _expert_loads[i].group_id = expert_group_size > 0 ? i / expert_group_size : 0;
            _expert_loads[i].estimated_load = 1.0;
            _expert_loads[i].replication_count = 1;
        }
    }

    void update_expert_loads(const std::vector<double>& new_loads) {
        std::lock_guard<std::mutex> lock(_mutex);
        for (size_t i = 0; i < new_loads.size() && i < _expert_loads.size(); ++i) {
            double alpha = 0.1;
            _expert_loads[i].estimated_load =
                alpha * new_loads[i] + (1.0 - alpha) * _expert_loads[i].estimated_load;
        }
    }

    void update_worker_heartbeat(int worker_id, const WorkerHeartbeat& hb) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (worker_id >= 0 && worker_id < _num_workers) {
            _worker_heartbeats[worker_id] = hb;
        }
    }

    std::vector<std::vector<int>> rebalance_experts() {
        std::lock_guard<std::mutex> lock(_mutex);

        bool use_hierarchical = (_expert_group_size > 0) &&
                                (_num_experts % _expert_group_size == 0) &&
                                (_num_workers % _num_nodes == 0);

        if (use_hierarchical) {
            return hierarchical_load_balance();
        } else {
            return global_load_balance();
        }
    }

    std::vector<int> get_experts_for_worker(int worker_id) const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<int> result;
        for (const auto& expert : _expert_loads) {
            for (int w : expert.assigned_workers) {
                if (w == worker_id) {
                    result.push_back(expert.expert_id);
                    break;
                }
            }
        }
        return result;
    }

    double get_worker_load(int worker_id) const {
        std::lock_guard<std::mutex> lock(_mutex);
        double total = 0;
        for (const auto& expert : _expert_loads) {
            for (int w : expert.assigned_workers) {
                if (w == worker_id) {
                    total += expert.estimated_load / expert.assigned_workers.size();
                    break;
                }
            }
        }
        return total;
    }

private:
    std::vector<std::vector<int>> hierarchical_load_balance() {
        int num_groups = _num_experts / _expert_group_size;
        int groups_per_node = num_groups / _num_nodes;

        std::vector<std::vector<int>> node_groups(_num_nodes);
        for (int g = 0; g < num_groups; ++g) {
            int node = g / groups_per_node;
            node_groups[node].push_back(g);
        }

        for (auto& expert : _expert_loads) {
            expert.assigned_workers.clear();
        }

        for (int node = 0; node < _num_nodes; ++node) {
            int node_start_worker = node * _workers_per_node;

            std::vector<int> node_experts;
            for (int group : node_groups[node]) {
                for (int e = group * _expert_group_size;
                     e < (group + 1) * _expert_group_size; ++e) {
                    node_experts.push_back(e);
                }
            }

            std::sort(node_experts.begin(), node_experts.end(),
                [this](int a, int b) {
                    return _expert_loads[a].estimated_load > _expert_loads[b].estimated_load;
                });

            std::vector<double> worker_loads(_workers_per_node, 0);
            for (int expert_id : node_experts) {
                int best_worker = 0;
                double min_load = worker_loads[0];
                for (int w = 1; w < _workers_per_node; ++w) {
                    if (worker_loads[w] < min_load) {
                        min_load = worker_loads[w];
                        best_worker = w;
                    }
                }

                int global_worker_id = node_start_worker + best_worker;
                _expert_loads[expert_id].assigned_workers.push_back(global_worker_id);
                worker_loads[best_worker] += _expert_loads[expert_id].estimated_load;
            }

            if (_num_redundant > 0) {
                assign_redundant_experts(node_experts, node_start_worker, worker_loads);
            }
        }

        return build_assignment_result();
    }

    std::vector<std::vector<int>> global_load_balance() {
        std::vector<int> expert_ids(_num_experts);
        std::iota(expert_ids.begin(), expert_ids.end(), 0);
        std::sort(expert_ids.begin(), expert_ids.end(),
            [this](int a, int b) {
                return _expert_loads[a].estimated_load > _expert_loads[b].estimated_load;
            });

        for (auto& expert : _expert_loads) {
            expert.assigned_workers.clear();
        }

        std::vector<double> worker_loads(_num_workers, 0);
        for (int expert_id : expert_ids) {
            int best_worker = 0;
            double min_load = worker_loads[0];
            for (int w = 1; w < _num_workers; ++w) {
                if (worker_loads[w] < min_load) {
                    min_load = worker_loads[w];
                    best_worker = w;
                }
            }

            _expert_loads[expert_id].assigned_workers.push_back(best_worker);
            worker_loads[best_worker] += _expert_loads[expert_id].estimated_load;
        }

        if (_num_redundant > 0) {
            assign_redundant_experts_global(expert_ids, worker_loads);
        }

        return build_assignment_result();
    }

    void assign_redundant_experts(const std::vector<int>& experts, int node_start,
                                   std::vector<double>& worker_loads) {
        int num_to_duplicate = std::min(_num_redundant, static_cast<int>(experts.size()));
        for (int i = 0; i < num_to_duplicate; ++i) {
            int expert_id = experts[i];
            int best_worker = 0;
            double min_load = worker_loads[0];
            for (int w = 1; w < _workers_per_node; ++w) {
                bool already_assigned = false;
                for (int w_id : _expert_loads[expert_id].assigned_workers) {
                    if (w_id == node_start + w) { already_assigned = true; break; }
                }
                if (!already_assigned && worker_loads[w] < min_load) {
                    min_load = worker_loads[w];
                    best_worker = w;
                }
            }
            int global_worker = node_start + best_worker;
            _expert_loads[expert_id].assigned_workers.push_back(global_worker);
            worker_loads[best_worker] += _expert_loads[expert_id].estimated_load / 2.0;
        }
    }

    void assign_redundant_experts_global(const std::vector<int>& experts,
                                          std::vector<double>& worker_loads) {
        int num_to_duplicate = std::min(_num_redundant, static_cast<int>(experts.size()));
        for (int i = 0; i < num_to_duplicate; ++i) {
            int expert_id = experts[i];
            int best_worker = 0;
            double min_load = worker_loads[0];
            for (int w = 1; w < _num_workers; ++w) {
                bool already_assigned = false;
                for (int w_id : _expert_loads[expert_id].assigned_workers) {
                    if (w_id == w) { already_assigned = true; break; }
                }
                if (!already_assigned && worker_loads[w] < min_load) {
                    min_load = worker_loads[w];
                    best_worker = w;
                }
            }
            _expert_loads[expert_id].assigned_workers.push_back(best_worker);
            worker_loads[best_worker] += _expert_loads[expert_id].estimated_load / 2.0;
        }
    }

    std::vector<std::vector<int>> build_assignment_result() {
        std::vector<std::vector<int>> result(_num_experts);
        for (const auto& expert : _expert_loads) {
            result[expert.expert_id] = expert.assigned_workers;
        }
        return result;
    }
};

// ============================================================================
// Load Balancing Strategy Enumeration
// ============================================================================
// All strategies use the heartbeat system as their foundation. They read
// effective_capacity_score, queue_depth, throughput, etc. from worker heartbeats
// to make scheduling decisions.

enum class LoadBalancingStrategy {
    HEARTBEAT_CAPACITY,       // Default: assign to worker with highest effective_capacity_score
    WEIGHTED_ROUND_ROBIN,     // Round-robin weighted by capacity score
    LEAST_CONNECTIONS,        // Assign to worker with fewest pending operations
    POWER_OF_TWO_CHOICES,     // Pick 2 random workers, assign to less loaded
    CONSISTENT_HASHING,       // Deterministic placement based on shard_id hash
    PREDICTIVE_LOAD,          // Uses EWMA of historical load to predict best worker
    MIN_MAX_FAIRNESS          // Minimizes the maximum load across all workers
};

// ============================================================================
// Pluggable Load Balancer Interface
// ============================================================================
// All strategies implement this interface. They all read from the same
// heartbeat data but use different algorithms to decide where to place work.

template<typename T>
class LoadBalancingStrategyImpl {
public:
    virtual ~LoadBalancingStrategyImpl() = default;

    virtual int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) = 0;

    virtual void update_state(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats) {
        (void)workers; (void)heartbeats;
    }
};

template<typename T>
class HeartbeatCapacityStrategy : public LoadBalancingStrategyImpl<T> {
public:
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)shard_id; (void)current_assignments;
        int best_worker = -1;
        double best_capacity = -1.0;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (!workers[i] || !workers[i]->is_healthy()) continue;
            if (i < heartbeats.size() && heartbeats[i].effective_capacity_score > best_capacity) {
                best_capacity = heartbeats[i].effective_capacity_score;
                best_worker = static_cast<int>(i);
            }
        }
        return best_worker;
    }
};

template<typename T>
class WeightedRoundRobinStrategy : public LoadBalancingStrategyImpl<T> {
private:
    int _current_index;
    std::vector<double> _weights;
public:
    WeightedRoundRobinStrategy() : _current_index(-1) {}
    void update_state(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats) override {
        _weights.resize(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            if (i < heartbeats.size() && workers[i] && workers[i]->is_healthy()) {
                _weights[i] = std::max(1.0, heartbeats[i].effective_capacity_score * 100.0);
            } else {
                _weights[i] = 0;
            }
        }
    }
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)shard_id; (void)current_assignments; (void)heartbeats;
        if (_weights.empty()) return -1;
        int n = workers.size();
        for (int attempt = 0; attempt < n; ++attempt) {
            _current_index = (_current_index + 1) % n;
            if (_weights[_current_index] > 0 &&
                workers[_current_index] && workers[_current_index]->is_healthy()) {
                return _current_index;
            }
        }
        return -1;
    }
};

template<typename T>
class LeastConnectionsStrategy : public LoadBalancingStrategyImpl<T> {
public:
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)shard_id;
        int best_worker = -1;
        double best_score = 1e18;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (!workers[i] || !workers[i]->is_healthy()) continue;
            double queue_depth = (i < heartbeats.size()) ? heartbeats[i].current_queue_depth : 0;
            int assigned = (i < current_assignments.size()) ? current_assignments[i] : 0;
            double score = queue_depth * 0.6 + assigned * 0.4;
            if (score < best_score) {
                best_score = score;
                best_worker = static_cast<int>(i);
            }
        }
        return best_worker;
    }
};

template<typename T>
class PowerOfTwoChoicesStrategy : public LoadBalancingStrategyImpl<T> {
private:
    mutable std::mt19937 _rng;
public:
    PowerOfTwoChoicesStrategy() : _rng(42) {}
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        std::vector<int> healthy;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (workers[i] && workers[i]->is_healthy()) {
                healthy.push_back(static_cast<int>(i));
            }
        }
        if (healthy.empty()) return -1;
        if (healthy.size() == 1) return healthy[0];
        std::uniform_int_distribution<int> dist(0, healthy.size() - 1);
        int idx1 = dist(_rng);
        int idx2;
        do { idx2 = dist(_rng); } while (idx2 == idx1);
        int w1 = healthy[idx1];
        int w2 = healthy[idx2];
        double load1 = (w1 < static_cast<int>(heartbeats.size())) ?
            heartbeats[w1].current_queue_depth + current_assignments[w1] : 0;
        double load2 = (w2 < static_cast<int>(heartbeats.size())) ?
            heartbeats[w2].current_queue_depth + current_assignments[w2] : 0;
        double cap1 = (w1 < static_cast<int>(heartbeats.size())) ?
            heartbeats[w1].effective_capacity_score : 0;
        double cap2 = (w2 < static_cast<int>(heartbeats.size())) ?
            heartbeats[w2].effective_capacity_score : 0;
        double norm1 = (cap1 > 0) ? load1 / cap1 : load1;
        double norm2 = (cap2 > 0) ? load2 / cap2 : load2;
        return (norm1 <= norm2) ? w1 : w2;
    }
};

template<typename T>
class ConsistentHashingStrategy : public LoadBalancingStrategyImpl<T> {
private:
    std::size_t hash_shard(std::size_t shard_id, int num_buckets) const {
        std::size_t h = shard_id * 0x9e3779b97f4a7c15ULL;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return h % num_buckets;
    }
public:
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)heartbeats; (void)current_assignments;
        std::vector<int> healthy;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (workers[i] && workers[i]->is_healthy()) {
                healthy.push_back(static_cast<int>(i));
            }
        }
        if (healthy.empty()) return -1;
        int idx = hash_shard(shard_id, healthy.size());
        return healthy[idx];
    }
};

template<typename T>
class PredictiveLoadStrategy : public LoadBalancingStrategyImpl<T> {
private:
    std::vector<double> _predicted_load;
    double _alpha;
public:
    PredictiveLoadStrategy(double alpha = 0.3) : _alpha(alpha) {}
    void update_state(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats) override {
        _predicted_load.resize(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            double current_load = 0;
            if (i < heartbeats.size()) {
                current_load = heartbeats[i].current_queue_depth +
                              (1.0 - heartbeats[i].effective_capacity_score) * 100;
            }
            if (_predicted_load[i] == 0) {
                _predicted_load[i] = current_load;
            } else {
                _predicted_load[i] = _alpha * current_load + (1.0 - _alpha) * _predicted_load[i];
            }
        }
    }
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)shard_id;
        int best_worker = -1;
        double best_score = 1e18;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (!workers[i] || !workers[i]->is_healthy()) continue;
            double predicted = (i < _predicted_load.size()) ? _predicted_load[i] : 0;
            int assigned = (i < current_assignments.size()) ? current_assignments[i] : 0;
            double cap = (i < heartbeats.size()) ? heartbeats[i].effective_capacity_score : 0;
            double score = (cap > 0) ? (predicted + assigned) / cap : (predicted + assigned);
            if (score < best_score) {
                best_score = score;
                best_worker = static_cast<int>(i);
            }
        }
        return best_worker;
    }
};

template<typename T>
class MinMaxFairnessStrategy : public LoadBalancingStrategyImpl<T> {
public:
    int select_worker(
        const std::vector<DistributedWorker<T>*>& workers,
        const std::vector<WorkerHeartbeat>& heartbeats,
        std::size_t shard_id,
        const std::vector<int>& current_assignments) override {
        (void)shard_id;
        int best_worker = -1;
        double min_max_load = 1e18;
        for (size_t i = 0; i < workers.size(); ++i) {
            if (!workers[i] || !workers[i]->is_healthy()) continue;
            double cap = (i < heartbeats.size()) ? heartbeats[i].effective_capacity_score : 0.001;
            int assigned = (i < current_assignments.size()) ? current_assignments[i] : 0;
            double queue = (i < heartbeats.size()) ? heartbeats[i].current_queue_depth : 0;
            double my_load = (queue + assigned + 1) / cap;
            double max_load = my_load;
            for (size_t j = 0; j < workers.size(); ++j) {
                if (j == i || !workers[j] || !workers[j]->is_healthy()) continue;
                double cap_j = (j < heartbeats.size()) ? heartbeats[j].effective_capacity_score : 0.001;
                int assigned_j = (j < current_assignments.size()) ? current_assignments[j] : 0;
                double queue_j = (j < heartbeats.size()) ? heartbeats[j].current_queue_depth : 0;
                double load_j = (queue_j + assigned_j) / cap_j;
                max_load = std::max(max_load, load_j);
            }
            if (max_load < min_max_load) {
                min_max_load = max_load;
                best_worker = static_cast<int>(i);
            }
        }
        return best_worker;
    }
};

template<typename T>
std::unique_ptr<LoadBalancingStrategyImpl<T>> create_load_balancing_strategy(
    LoadBalancingStrategy strategy) {
    switch (strategy) {
        case LoadBalancingStrategy::HEARTBEAT_CAPACITY:
            return std::make_unique<HeartbeatCapacityStrategy<T>>();
        case LoadBalancingStrategy::WEIGHTED_ROUND_ROBIN:
            return std::make_unique<WeightedRoundRobinStrategy<T>>();
        case LoadBalancingStrategy::LEAST_CONNECTIONS:
            return std::make_unique<LeastConnectionsStrategy<T>>();
        case LoadBalancingStrategy::POWER_OF_TWO_CHOICES:
            return std::make_unique<PowerOfTwoChoicesStrategy<T>>();
        case LoadBalancingStrategy::CONSISTENT_HASHING:
            return std::make_unique<ConsistentHashingStrategy<T>>();
        case LoadBalancingStrategy::PREDICTIVE_LOAD:
            return std::make_unique<PredictiveLoadStrategy<T>>();
        case LoadBalancingStrategy::MIN_MAX_FAIRNESS:
            return std::make_unique<MinMaxFairnessStrategy<T>>();
        default:
            return std::make_unique<HeartbeatCapacityStrategy<T>>();
    }
}

// ============================================================================
// Distributed Load Balancer - USES HEARTBEATS AS ITS FOUNDATION
// ============================================================================
// The load balancer reads effective_capacity_score from each worker's heartbeat
// and assigns shards to the worker with the highest available capacity.

template<typename T>
class DistributedLoadBalancer {
private:
    std::vector<DistributedWorker<T>*> _workers;
    std::vector<WorkerHeartbeat> _heartbeats;
    std::unique_ptr<ExpertLoadBalancer> _expert_balancer;
    std::unique_ptr<LoadBalancingStrategyImpl<T>> _lb_strategy;
    mutable std::mutex _mutex;
    std::atomic<bool> _running;
    std::thread _heartbeat_monitor;

    double _heartbeat_interval_seconds;
    double _worker_timeout_seconds;

    std::unordered_map<std::size_t, int> _shard_assignments;
    std::vector<int> _assignment_counts;  // Track shards per worker for strategies

public:
    DistributedLoadBalancer(double heartbeat_interval = 1.0, double timeout = 30.0,
                            LoadBalancingStrategy strategy = LoadBalancingStrategy::HEARTBEAT_CAPACITY)
        : _running(false), _heartbeat_interval_seconds(heartbeat_interval),
          _worker_timeout_seconds(timeout) {
        _lb_strategy = create_load_balancing_strategy<T>(strategy);
    }

    // Set load balancing strategy (can be changed at runtime)
    void set_strategy(LoadBalancingStrategy strategy) {
        std::lock_guard<std::mutex> lock(_mutex);
        _lb_strategy = create_load_balancing_strategy<T>(strategy);
    }

    // Set custom strategy implementation (for user-defined strategies)
    void set_custom_strategy(std::unique_ptr<LoadBalancingStrategyImpl<T>> strategy) {
        std::lock_guard<std::mutex> lock(_mutex);
        _lb_strategy = std::move(strategy);
    }

    ~DistributedLoadBalancer() {
        stop();
    }

    void register_worker(DistributedWorker<T>* worker) {
        std::lock_guard<std::mutex> lock(_mutex);
        _workers.push_back(worker);
        _heartbeats.resize(_workers.size());
        _assignment_counts.resize(_workers.size(), 0);
    }

    void set_expert_balancer(std::unique_ptr<ExpertLoadBalancer> balancer) {
        _expert_balancer = std::move(balancer);
    }

    void start() {
        _running.store(true);
        _heartbeat_monitor = std::thread([this]() {
            while (_running.load()) {
                monitor_heartbeats();
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(_heartbeat_interval_seconds * 1000)));
            }
        });
    }

    void stop() {
        _running.store(false);
        if (_heartbeat_monitor.joinable()) {
            _heartbeat_monitor.join();
        }
    }

    // === STRATEGY-BASED SHARD ASSIGNMENT ===
    // Uses the configured load balancing strategy to decide where to place work.
    // All strategies read from the same heartbeat data but use different algorithms.
    int assign_shard(std::size_t shard_id) {
        std::lock_guard<std::mutex> lock(_mutex);

        // Check if already assigned to a healthy worker
        auto it = _shard_assignments.find(shard_id);
        if (it != _shard_assignments.end()) {
            int worker_id = it->second;
            if (worker_id < static_cast<int>(_workers.size()) &&
                _workers[worker_id] && _workers[worker_id]->is_healthy()) {
                return worker_id;
            }
        }

        // Update strategy state with latest heartbeats
        if (_lb_strategy) {
            _lb_strategy->update_state(_workers, _heartbeats);
        }

        // Use strategy to select worker
        int selected = -1;
        if (_lb_strategy) {
            selected = _lb_strategy->select_worker(
                _workers, _heartbeats, shard_id, _assignment_counts);
        }

        if (selected >= 0) {
            _shard_assignments[shard_id] = selected;
            if (selected < static_cast<int>(_assignment_counts.size())) {
                _assignment_counts[selected]++;
            }
        }

        return selected;
    }

    std::vector<int> get_healthy_workers() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<int> result;
        for (size_t i = 0; i < _workers.size(); ++i) {
            if (_workers[i] && _workers[i]->is_healthy()) {
                result.push_back(static_cast<int>(i));
            }
        }
        return result;
    }

    WorkerHeartbeat get_worker_heartbeat(int worker_id) const {
        std::lock_guard<std::mutex> lock(_mutex);
        if (worker_id >= 0 && worker_id < static_cast<int>(_heartbeats.size())) {
            return _heartbeats[worker_id];
        }
        return WorkerHeartbeat();
    }

    std::vector<int> get_experts_for_worker(int worker_id) const {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_expert_balancer) {
            return _expert_balancer->get_experts_for_worker(worker_id);
        }
        return {};
    }

    // Get current assignment counts per worker (for monitoring)
    std::vector<int> get_assignment_counts() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _assignment_counts;
    }

    // Get load distribution stats
    std::string get_load_distribution_string() const {
        std::lock_guard<std::mutex> lock(_mutex);
        std::ostringstream oss;
        oss << "Load distribution: ";
        for (size_t i = 0; i < _workers.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "W" << i << "=" << _assignment_counts[i];
        }
        return oss.str();
    }

private:
    void monitor_heartbeats() {
        std::lock_guard<std::mutex> lock(_mutex);
        for (size_t i = 0; i < _workers.size(); ++i) {
            if (!_workers[i]) continue;

            auto hb = _workers[i]->get_heartbeat();
            _heartbeats[i] = hb;

            if (_expert_balancer) {
                _expert_balancer->update_worker_heartbeat(static_cast<int>(i), hb);
            }

            if (!hb.is_healthy(_worker_timeout_seconds)) {
                _workers[i]->mark_failed();
            }
        }
    }
};

// ============================================================================
// Worker Coordinator: Sets up mailboxes between heterogeneous workers
// ============================================================================

template<typename T>
class WorkerCoordinator {
private:
    std::vector<DistributedWorker<T>*> _workers;
    std::vector<std::vector<std::unique_ptr<TensorMailbox<T>>>> _mailboxes;

public:
    void register_workers(const std::vector<DistributedWorker<T>*>& workers) {
        _workers = workers;
        int n = workers.size();

        _mailboxes.resize(n);
        for (int i = 0; i < n; ++i) {
            _mailboxes[i].resize(n);
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    _mailboxes[i][j] = std::make_unique<TensorMailbox<T>>();
                }
            }
        }
    }

    TensorMailbox<T>* get_mailbox(int receiver, int sender) {
        if (receiver >= 0 && receiver < static_cast<int>(_mailboxes.size()) &&
            sender >= 0 && sender < static_cast<int>(_mailboxes[receiver].size())) {
            return _mailboxes[receiver][sender].get();
        }
        return nullptr;
    }

    void barrier() {
        if (!_workers.empty()) {
            _workers[0]->barrier();
        }
    }
};

// ============================================================================
// Data Parallel Executor (works with ANY backend mix)
// ============================================================================

template<typename T>
class DataParallelExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    WorkerCoordinator<T>* _coordinator;

public:
    DataParallelExecutor(const std::vector<DistributedWorker<T>*>& workers,
                         WorkerCoordinator<T>* coord = nullptr)
        : _workers(workers), _coordinator(coord) {}

    std::vector<std::unique_ptr<TensorBase<T>>> forward_all(
        const std::vector<const TensorBase<T>*>& data_shards) {

        std::vector<std::future<std::unique_ptr<TensorBase<T>>>> futures;
        futures.reserve(_workers.size());

        for (size_t i = 0; i < _workers.size() && i < data_shards.size(); ++i) {
            auto* worker = _workers[i];
            auto* shard = data_shards[i];
            futures.push_back(std::async(std::launch::async, [worker, shard]() {
                return worker->forward_shard(shard);
            }));
        }

        std::vector<std::unique_ptr<TensorBase<T>>> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }
        return results;
    }

    void backward_all(const std::vector<const TensorBase<T>*>& grad_shards) {
        std::vector<std::future<void>> futures;
        futures.reserve(_workers.size());

        for (size_t i = 0; i < _workers.size() && i < grad_shards.size(); ++i) {
            auto* worker = _workers[i];
            auto* shard = grad_shards[i];
            futures.push_back(std::async(std::launch::async, [worker, shard]() {
                worker->backward_shard(shard);
            }));
        }

        for (auto& f : futures) {
            f.get();
        }

        for (auto* worker : _workers) {
            worker->sync_gradients();
        }
    }

    void step_all(float lr) {
        for (auto* worker : _workers) {
            worker->sgd_step(lr);
            worker->zero_grad();
        }
    }
};

// ============================================================================
// Tensor Parallel Executor (works with ANY backend mix)
// ============================================================================

template<typename T>
class TensorParallelExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    int _tp_size;
    WorkerCoordinator<T>* _coordinator;

public:
    TensorParallelExecutor(const std::vector<DistributedWorker<T>*>& workers,
                           int tp_size, WorkerCoordinator<T>* coord = nullptr)
        : _workers(workers), _tp_size(tp_size), _coordinator(coord) {}

    std::unique_ptr<TensorBase<T>> column_parallel_forward(
        const TensorBase<T>* input) {

        std::vector<std::future<std::unique_ptr<TensorBase<T>>>> futures;
        futures.reserve(_workers.size());

        for (auto* worker : _workers) {
            futures.push_back(std::async(std::launch::async, [worker, input]() {
                return worker->forward_shard(input);
            }));
        }

        // Get all results first
        std::vector<std::unique_ptr<TensorBase<T>>> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        std::size_t shard_size = results[0]->total_size();
        std::size_t total_size = shard_size * _tp_size;
        std::vector<std::size_t> out_shape = {total_size};
        auto gathered = std::make_unique<DenseTensor<T>>(out_shape.data(), out_shape.size());

        for (size_t i = 0; i < results.size(); ++i) {
            for (std::size_t j = 0; j < results[i]->total_size(); ++j) {
                gathered->set_element(i * shard_size + j, results[i]->get_element(j));
            }
        }

        return gathered;
    }

    std::unique_ptr<TensorBase<T>> row_parallel_forward(
        const TensorBase<T>* input) {

        std::vector<std::future<std::unique_ptr<TensorBase<T>>>> futures;
        futures.reserve(_workers.size());

        for (auto* worker : _workers) {
            futures.push_back(std::async(std::launch::async, [worker, input]() {
                return worker->forward_shard(input);
            }));
        }

        // Get all results first
        std::vector<std::unique_ptr<TensorBase<T>>> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        std::size_t shard_size = results[0]->total_size();
        std::vector<std::size_t> out_shape = {shard_size};
        auto reduced = std::make_unique<DenseTensor<T>>(out_shape.data(), out_shape.size());

        for (std::size_t j = 0; j < shard_size; ++j) {
            T sum = T{};
            for (const auto& r : results) {
                sum += r->get_element(j);
            }
            reduced->set_element(j, sum / static_cast<T>(_tp_size));
        }

        return reduced;
    }

    std::unique_ptr<TensorBase<T>> forward(
        const TensorBase<T>* input,
        const std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*)>& col_op,
        const std::function<std::unique_ptr<TensorBase<T>>(const TensorBase<T>*)>& row_op) {

        (void)col_op; (void)row_op;

        auto col_result = column_parallel_forward(input);
        auto final_result = row_parallel_forward(col_result.get());
        return final_result;
    }
};

// ============================================================================
// Pipeline Parallel Executor (works with ANY backend mix)
// ============================================================================

template<typename T>
class PipelineParallelExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    int _pp_size;
    int _num_micro_batches;

public:
    PipelineParallelExecutor(const std::vector<DistributedWorker<T>*>& workers,
                             int pp_size, int num_micro_batches = 4)
        : _workers(workers), _pp_size(pp_size), _num_micro_batches(num_micro_batches) {}

    std::unique_ptr<TensorBase<T>> forward_pipeline(
        const TensorBase<T>* input) {

        auto micro_batches = split_into_micro_batches(input, _num_micro_batches);

        std::vector<std::unique_ptr<TensorBase<T>>> stage_outputs;
        stage_outputs.reserve(_pp_size);

        // Warmup: fill the pipeline
        for (int step = 0; step < _pp_size - 1 && step < static_cast<int>(micro_batches.size()); ++step) {
            auto* current_input = micro_batches[step].get();
            stage_outputs.push_back(_workers[step]->forward_shard(current_input));
        }

        std::vector<std::unique_ptr<TensorBase<T>>> final_outputs;
        for (int mb = 0; mb < _num_micro_batches && mb < static_cast<int>(micro_batches.size()); ++mb) {
            std::unique_ptr<TensorBase<T>> current_ptr;
            std::vector<std::size_t> shape;
            auto* s = micro_batches[mb]->shape();
            for (std::size_t i = 0; i < micro_batches[mb]->ndim(); ++i) shape.push_back(s[i]);
            current_ptr = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t j = 0; j < micro_batches[mb]->total_size(); ++j) {
                current_ptr->set_element(j, micro_batches[mb]->get_element(j));
            }

            // Pass through pipeline stages
            for (int stage = 0; stage < _pp_size; ++stage) {
                current_ptr = _workers[stage]->forward_shard(current_ptr.get());
            }

            final_outputs.push_back(std::move(current_ptr));
        }

        return concatenate_tensors(final_outputs);
    }

    void backward_pipeline(const TensorBase<T>* grad_output) {
        auto micro_grads = split_into_micro_batches(grad_output, _num_micro_batches);

        for (int mb = 0; mb < _num_micro_batches; ++mb) {
            std::unique_ptr<TensorBase<T>> current_ptr;
            std::vector<std::size_t> shape;
            auto* s = micro_grads[mb]->shape();
            for (std::size_t i = 0; i < micro_grads[mb]->ndim(); ++i) shape.push_back(s[i]);
            current_ptr = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t j = 0; j < micro_grads[mb]->total_size(); ++j) {
                current_ptr->set_element(j, micro_grads[mb]->get_element(j));
            }

            for (int stage = _pp_size - 1; stage >= 0; --stage) {
                current_ptr = _workers[stage]->backward_shard(current_ptr.get());
            }
        }
    }

private:
    std::vector<std::unique_ptr<TensorBase<T>>> split_into_micro_batches(
        const TensorBase<T>* input, int num_batches) {

        std::vector<std::unique_ptr<TensorBase<T>>> batches;
        std::size_t total = input->total_size();
        std::size_t batch_size = (total + num_batches - 1) / num_batches;

        for (int i = 0; i < num_batches; ++i) {
            std::size_t offset = i * batch_size;
            std::size_t count = (offset < total) ? std::min(batch_size, total - offset) : 0;
            if (count == 0) break;

            std::vector<std::size_t> shape = {count};
            auto batch = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t j = 0; j < count; ++j) {
                batch->set_element(j, input->get_element(offset + j));
            }
            batches.push_back(std::move(batch));
        }

        return batches;
    }

    std::unique_ptr<TensorBase<T>> concatenate_tensors(
        const std::vector<std::unique_ptr<TensorBase<T>>>& tensors) {

        std::size_t total = 0;
        for (const auto& t : tensors) total += t->total_size();

        std::vector<std::size_t> shape = {total};
        auto result = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());

        std::size_t offset = 0;
        for (const auto& t : tensors) {
            for (std::size_t i = 0; i < t->total_size(); ++i) {
                result->set_element(offset + i, t->get_element(i));
            }
            offset += t->total_size();
        }
        return result;
    }
};

// ============================================================================
// FSDP Executor (works with ANY backend mix)
// ============================================================================

template<typename T>
class FSDPExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    int _dp_size;
    WorkerCoordinator<T>* _coordinator;

public:
    FSDPExecutor(const std::vector<DistributedWorker<T>*>& workers,
                 int dp_size, WorkerCoordinator<T>* coord = nullptr)
        : _workers(workers), _dp_size(dp_size), _coordinator(coord) {}

    std::vector<std::unique_ptr<TensorBase<T>>> forward_all(
        const std::vector<const TensorBase<T>*>& data_shards,
        const TensorBase<T>* param_shards) {

        std::vector<std::unique_ptr<TensorBase<T>>> results;
        results.reserve(_workers.size());

        std::size_t shard_size = param_shards->total_size();
        std::size_t full_size = shard_size * _dp_size;
        std::vector<std::size_t> full_shape = {full_size};
        auto full_params = std::make_unique<DenseTensor<T>>(full_shape.data(), full_shape.size());

        for (size_t i = 0; i < _workers.size(); ++i) {
            _workers[i]->allgather(param_shards, full_params.get());
        }

        for (size_t i = 0; i < _workers.size() && i < data_shards.size(); ++i) {
            auto* worker = _workers[i];
            worker->sync_parameters();
            auto result = worker->forward_shard(data_shards[i]);
            results.push_back(std::move(result));
        }

        return results;
    }

    void backward_all(const std::vector<const TensorBase<T>*>& grad_shards) {
        for (size_t i = 0; i < _workers.size() && i < grad_shards.size(); ++i) {
            _workers[i]->backward_shard(grad_shards[i]);
        }

        for (auto* worker : _workers) {
            worker->sync_gradients();
        }
    }

    void step_all(float lr) {
        for (auto* worker : _workers) {
            worker->sgd_step(lr);
            worker->zero_grad();
        }
    }
};

// ============================================================================
// Expert Parallel Executor (MoE, works with ANY backend mix)
// ============================================================================

template<typename T>
class ExpertParallelExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    std::unique_ptr<ExpertLoadBalancer> _balancer;
    std::vector<std::vector<int>> _expert_assignments;
    std::vector<std::unique_ptr<TensorBase<T>>> _expert_params;

public:
    ExpertParallelExecutor(const std::vector<DistributedWorker<T>*>& workers,
                           std::unique_ptr<ExpertLoadBalancer> balancer)
        : _workers(workers), _balancer(std::move(balancer)) {

        _expert_assignments = _balancer->rebalance_experts();
        int num_experts = _expert_assignments.size();

        _expert_params.resize(num_experts);
        for (int i = 0; i < num_experts; ++i) {
            std::size_t expert_size = 64;
            std::vector<std::size_t> shape = {expert_size};

            BackendType backend = BackendType::CPU_DENSE;
            if (!_expert_assignments[i].empty()) {
                int w = _expert_assignments[i][0];
                if (w < static_cast<int>(_workers.size())) {
                    backend = _workers[w]->backend_type();
                }
            }

            _expert_params[i] = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());
            for (std::size_t j = 0; j < expert_size; ++j) {
                _expert_params[i]->set_element(j, static_cast<T>(0.01f * (j % 10)));
            }
        }
    }

    std::vector<std::unique_ptr<TensorBase<T>>> route_and_compute(
        const TensorBase<T>* input, int num_experts_per_token = 2) {

        (void)num_experts_per_token;
        std::vector<std::unique_ptr<TensorBase<T>>> outputs;

        for (size_t w = 0; w < _workers.size(); ++w) {
            auto worker_experts = _balancer->get_experts_for_worker(static_cast<int>(w));

            if (worker_experts.empty()) continue;

            for (int expert_id : worker_experts) {
                if (expert_id < static_cast<int>(_expert_params.size())) {
                    _workers[w]->sync_parameters();
                    auto output = _workers[w]->forward_shard(input);
                    outputs.push_back(std::move(output));
                }
            }
        }

        return outputs;
    }

    void rebalance() {
        _expert_assignments = _balancer->rebalance_experts();
    }

    void update_expert_loads(const std::vector<double>& loads) {
        if (_balancer) {
            _balancer->update_expert_loads(loads);
        }
    }
};

// ============================================================================
// 3D Parallel Executor (TP + FSDP + PP combined, works with ANY backend mix)
// ============================================================================

template<typename T>
class Parallel3DExecutor {
private:
    std::vector<DistributedWorker<T>*> _workers;
    int _tp_size;
    int _dp_size;
    int _pp_size;
    WorkerCoordinator<T>* _coordinator;

public:
    Parallel3DExecutor(const std::vector<DistributedWorker<T>*>& workers,
                       int tp, int dp, int pp, WorkerCoordinator<T>* coord = nullptr)
        : _workers(workers), _tp_size(tp), _dp_size(dp), _pp_size(pp),
          _coordinator(coord) {}

    std::vector<DistributedWorker<T>*> get_tp_group(int dp_rank, int pp_rank) {
        std::vector<DistributedWorker<T>*> group;
        for (int tp = 0; tp < _tp_size; ++tp) {
            int rank = pp_rank * (_tp_size * _dp_size) + dp_rank * _tp_size + tp;
            if (rank < static_cast<int>(_workers.size())) {
                group.push_back(_workers[rank]);
            }
        }
        return group;
    }

    std::vector<DistributedWorker<T>*> get_dp_group(int tp_rank, int pp_rank) {
        std::vector<DistributedWorker<T>*> group;
        for (int dp = 0; dp < _dp_size; ++dp) {
            int rank = pp_rank * (_tp_size * _dp_size) + dp * _tp_size + tp_rank;
            if (rank < static_cast<int>(_workers.size())) {
                group.push_back(_workers[rank]);
            }
        }
        return group;
    }

    std::vector<DistributedWorker<T>*> get_pp_group(int tp_rank, int dp_rank) {
        std::vector<DistributedWorker<T>*> group;
        for (int pp = 0; pp < _pp_size; ++pp) {
            int rank = pp * (_tp_size * _dp_size) + dp_rank * _tp_size + tp_rank;
            if (rank < static_cast<int>(_workers.size())) {
                group.push_back(_workers[rank]);
            }
        }
        return group;
    }

    std::unique_ptr<TensorBase<T>> forward_3d(const TensorBase<T>* input) {
        PipelineParallelExecutor<T> pp_exec(_workers, _pp_size);
        return pp_exec.forward_pipeline(input);
    }

    void backward_3d(const TensorBase<T>* grad_output) {
        PipelineParallelExecutor<T> pp_exec(_workers, _pp_size);
        pp_exec.backward_pipeline(grad_output);
    }
};

// ============================================================================
// Parallelism Factory
// ============================================================================

template<typename T>
class ParallelismFactory {
public:
    static std::unique_ptr<DataParallelExecutor<T>> create_data_parallel(
        const std::vector<DistributedWorker<T>*>& workers,
        WorkerCoordinator<T>* coord = nullptr) {
        return std::make_unique<DataParallelExecutor<T>>(workers, coord);
    }

    static std::unique_ptr<TensorParallelExecutor<T>> create_tensor_parallel(
        const std::vector<DistributedWorker<T>*>& workers, int tp_size,
        WorkerCoordinator<T>* coord = nullptr) {
        return std::make_unique<TensorParallelExecutor<T>>(workers, tp_size, coord);
    }

    static std::unique_ptr<PipelineParallelExecutor<T>> create_pipeline_parallel(
        const std::vector<DistributedWorker<T>*>& workers, int pp_size,
        int num_micro_batches = 4, WorkerCoordinator<T>* coord = nullptr) {
        (void)coord;
        return std::make_unique<PipelineParallelExecutor<T>>(workers, pp_size, num_micro_batches);
    }

    static std::unique_ptr<FSDPExecutor<T>> create_fsdp(
        const std::vector<DistributedWorker<T>*>& workers, int dp_size,
        WorkerCoordinator<T>* coord = nullptr) {
        return std::make_unique<FSDPExecutor<T>>(workers, dp_size, coord);
    }

    static std::unique_ptr<ExpertParallelExecutor<T>> create_expert_parallel(
        const std::vector<DistributedWorker<T>*>& workers,
        std::unique_ptr<ExpertLoadBalancer> balancer) {
        return std::make_unique<ExpertParallelExecutor<T>>(workers, std::move(balancer));
    }

    static std::unique_ptr<Parallel3DExecutor<T>> create_3d_parallel(
        const std::vector<DistributedWorker<T>*>& workers,
        int tp, int dp, int pp, WorkerCoordinator<T>* coord = nullptr) {
        return std::make_unique<Parallel3DExecutor<T>>(workers, tp, dp, pp, coord);
    }

    static std::unique_ptr<ExpertLoadBalancer> create_expert_balancer(
        int num_experts, int num_workers, int num_nodes, int workers_per_node,
        int num_redundant = 0, int expert_group_size = 0) {
        return std::make_unique<ExpertLoadBalancer>(
            num_experts, num_workers, num_nodes, workers_per_node,
            num_redundant, expert_group_size);
    }
};

// ============================================================================
// Shard Operations
// ============================================================================

template<typename T>
std::unique_ptr<TensorBase<T>> extract_shard(const TensorBase<T>* tensor,
                                              const ShardDescriptor& desc) {
    std::vector<std::size_t> shape = {desc.count};
    auto shard = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());

    for (std::size_t i = 0; i < desc.count; ++i) {
        shard->set_element(i, tensor->get_element(desc.offset + i));
    }
    return shard;
}

template<typename T>
std::unique_ptr<TensorBase<T>> assemble_shards(
    const std::vector<const TensorBase<T>*>& shards,
    const std::vector<ShardDescriptor>& descriptors) {

    std::size_t total = 0;
    for (const auto& d : descriptors) total += d.count;

    std::vector<std::size_t> shape = {total};
    auto result = std::make_unique<DenseTensor<T>>(shape.data(), shape.size());

    for (size_t i = 0; i < shards.size(); ++i) {
        const auto& desc = descriptors[i];
        for (std::size_t j = 0; j < desc.count; ++j) {
            result->set_element(desc.offset + j, shards[i]->get_element(j));
        }
    }
    return result;
}
