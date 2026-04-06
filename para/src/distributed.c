#include <mpi.h>
#include <unistd.h>
#include "data.c"
#include "model.c"


#define rank0_printf(rank, ...) if (rank == 0) { printf(__VA_ARGS__); }


int max(int a, int b) {
    return a >= b ? a : b;
}


// ========== Communication utils ==========

typedef struct {
    int tp_rank;
    int tp_size;
    MPI_Group tp_group;
    MPI_Comm tp_comm;

    int dp_rank;
    int dp_size;
    MPI_Group dp_group;
    MPI_Comm dp_comm;

    int pp_rank;
    int pp_size;
    MPI_Group pp_group;
    MPI_Comm pp_comm;

    int world_rank;
    int world_size; 
} Dist;



Dist* Dist_create(int tp_size, int dp_size, int pp_size) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (tp_size * dp_size * pp_size != world_size) {
        rank0_printf(
            world_rank, 
            "Invalid distributed environment: tp=%d * dp%d * pp=%d != world_size=%d\n", 
            tp_size, dp_size, pp_size, world_size
        );
        MPI_Finalize();
        exit(0);
    }
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group tp_group = NULL; MPI_Comm tp_comm = NULL;
    MPI_Group dp_group = NULL; MPI_Comm dp_comm = NULL;
    MPI_Group pp_group = NULL; MPI_Comm pp_comm = NULL;

    // Derive process group rank.
    int tp_rank = world_rank % tp_size;
    int dp_rank = (world_rank / tp_size) % dp_size;
    int pp_rank = world_rank / (tp_size * dp_size);

    if (tp_size > 1) {
        int tp_group_ranks[tp_size];
        for (int i = 0; i < tp_size; i++) {
            tp_group_ranks[i] = pp_rank * (tp_size * dp_size) + dp_rank * tp_size + i;
        }
        tp_group = malloc(sizeof(MPI_Group)); tp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, tp_size, tp_group_ranks, &tp_group);
        MPI_Comm_create(MPI_COMM_WORLD, tp_group, &tp_comm);
   }

    if (dp_size > 1) {
        int dp_group_ranks[dp_size];
        for (int i = 0; i < dp_size; i++) {
            dp_group_ranks[i] = pp_rank * (tp_size * dp_size) + i * tp_size + tp_rank;
        }
        dp_group = malloc(sizeof(MPI_Group)); dp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, dp_size, dp_group_ranks, &dp_group);
        MPI_Comm_create(MPI_COMM_WORLD, dp_group, &dp_comm);  
    }

    if (pp_size > 1) {
        int pp_group_ranks[pp_size];
        for (int i = 0; i < pp_size; i++) {
            pp_group_ranks[i] = i * (tp_size * dp_size) + dp_rank * tp_size + tp_rank;
        }
        pp_group = malloc(sizeof(MPI_Group)); pp_comm = malloc(sizeof(MPI_Comm));
        MPI_Group_incl(world_group, pp_size, pp_group_ranks, &pp_group);
        MPI_Comm_create(MPI_COMM_WORLD, pp_group, &pp_comm);
    }

    Dist* self = malloc(sizeof(Dist));
    self->tp_rank = tp_rank;
    self->tp_size = tp_size;
    self->tp_group = tp_group;
    self->tp_comm = tp_comm;
    self->dp_rank = dp_rank;
    self->dp_size = dp_size;
    self->dp_group = dp_group;
    self->dp_comm = dp_comm;
    self->pp_rank = pp_rank;
    self->pp_size = pp_size;
    self->pp_group = pp_group;
    self->pp_comm = pp_comm;
    self->world_rank = world_rank;
    self->world_size = world_size;
    return self;
}


static void send(float* input, int input_size, int to_rank, MPI_Comm pg_comm) {
    MPI_Send(input, input_size, MPI_FLOAT, to_rank, 0, pg_comm);
}


static void recv(float* output, int output_size, int from_rank, MPI_Comm pg_comm) {
    MPI_Status status;
    MPI_Recv(output, output_size, MPI_FLOAT, from_rank, 0, pg_comm, &status);
}


void allgather(float* shard, int shard_size, float* full, MPI_Comm pg_comm) {
    MPI_Allgather(
        shard, shard_size, MPI_FLOAT, full, shard_size, MPI_FLOAT, pg_comm 
    );
}


void reducescatter_mean(
    float* full, float* shard, int shard_size, MPI_Comm pg_comm, int pg_size
) {
    int shard_sizes[pg_size];
    for (int i = 0; i < pg_size; i++) {
        shard_sizes[i] = shard_size;
    }
    MPI_Reduce_scatter(full, shard, shard_sizes, MPI_FLOAT, MPI_SUM, pg_comm);
    for (int i = 0; i < shard_size; i++) {
        shard[i] = shard[i] / pg_size;
    }
}


void allreduce_mean(float* input, int size, MPI_Comm pg_comm, int pg_size) {
    MPI_Allreduce(MPI_IN_PLACE, input, size, MPI_FLOAT, MPI_SUM, pg_comm);
    for (int i = 0; i < size; i++) {
        input[i] = input[i] / pg_size;
    }
}


// ========== Data loader utils ==========

// TODO(eugen): Consider adding this functionality directly into Dataset_get_batch so 
// we can get the rank batch directly without first getting the global batch.
void Dataset_get_rank_batch(
    Dataset* self,
    int* global_Xs, 
    int* global_Ys, 
    int* Xs, 
    int* Ys, 
    int global_batch_size, 
    int rank,
    int world_size
) {
    Dataset_get_batch(self, global_Xs, global_Ys, global_batch_size);
    int local_b = 0;
    for (int b = 0; b < global_batch_size; b++) {
        if (b % world_size != rank) {
            continue;
        }

        for (int i = 0; i < self->seq_len; i ++) {
            int local_idx = local_b * self->seq_len + i;
            int global_idx = b * self->seq_len + i;
            Xs[local_idx] = global_Xs[global_idx];
        }
        Ys[local_b] = global_Ys[b];
        local_b += 1;
    }
}


// ========== Tensor parallelism utils ==========

void Model_shard_tp(Model* self, int pg_rank, int pg_size) {
    // Shard fc_1 to be column parallel across ranks.
    int fc_1_shard_cols = self->fc_1->out_features / pg_size;
    float* fc_1_weight_shard = malloc(sizeof(float) * self->fc_1->in_features * fc_1_shard_cols);
    float* fc_1_d_weight_shard = calloc(sizeof(float), self->fc_1->in_features * fc_1_shard_cols);
    for (int row = 0; row < self->fc_1->in_features; row++) {
        int shard_offset = row * fc_1_shard_cols;
        int weight_offset = row * self->fc_1->out_features + pg_rank * fc_1_shard_cols;
        memcpy(fc_1_weight_shard + shard_offset, self->fc_1->weight + weight_offset, sizeof(float) * fc_1_shard_cols);
    }
    float* fc_1_bias_shard = malloc(sizeof(float) * fc_1_shard_cols);
    float* fc_1_d_bias_shard = calloc(sizeof(float), fc_1_shard_cols);
    memcpy(fc_1_bias_shard, self->fc_1->bias + pg_rank * fc_1_shard_cols, sizeof(float) * fc_1_shard_cols);
    free(self->fc_1->weight); self->fc_1->weight = fc_1_weight_shard;
    free(self->fc_1->d_weight); self->fc_1->d_weight = fc_1_d_weight_shard;
    free(self->fc_1->bias); self->fc_1->bias = fc_1_bias_shard;
    free(self->fc_1->d_bias); self->fc_1->d_bias = fc_1_d_bias_shard;
    self->fc_1->out_features = fc_1_shard_cols;

    // Shard fc_2 to be row parallel.
    int fc_2_shard_rows = self->fc_2->in_features / pg_size;
    int fc_2_weight_shard_size = Linear_weight_numel(self->fc_2) / pg_size;
    float* fc_2_weight_shard = malloc(sizeof(float) * fc_2_weight_shard_size);
    float* fc_2_d_weight_shard = calloc(sizeof(float), fc_2_weight_shard_size);
    memcpy(fc_2_weight_shard, self->fc_2->weight + pg_rank * fc_2_weight_shard_size, sizeof(float) * fc_2_weight_shard_size);
    free(self->fc_2->weight); self->fc_2->weight = fc_2_weight_shard;
    free(self->fc_2->d_weight); self->fc_2->d_weight = fc_2_d_weight_shard;
    self->fc_2->in_features = fc_2_shard_rows;

    // Update activation shapes to match.
    Activation* fc_1_out_shard = Activation_create(self->fc_1_out->batch_size, fc_1_shard_cols);
    Activation* relu_out_shard = Activation_create(self->relu_out->batch_size, fc_1_shard_cols);
    Activation_destory(self->fc_1_out); self->fc_1_out = fc_1_out_shard;
    Activation_destory(self->relu_out); self->relu_out = relu_out_shard;
}


// ========== Fully-sharded data parallelism utils ==========

void Model_pad_vocab_fsdp(Model* self, int pg_size) {
    int vocab_size_padded = self->wte->vocab_size + (pg_size - (self->wte->vocab_size % pg_size));
    float* wte_padded = calloc(sizeof(float), vocab_size_padded * self->wte->emb_size);
    memcpy(wte_padded, self->wte->embedding, sizeof(float) * Embedding_numel(self->wte));
    float* wte_d_padded = calloc(sizeof(float), vocab_size_padded * self->wte->emb_size);
    free(self->wte->embedding); self->wte->embedding = wte_padded;
    free(self->wte->d_embedding); self->wte->d_embedding = wte_d_padded;
    self->wte->vocab_size = vocab_size_padded;
}


float* Model_create_flat_buffer_fsdp(Model* self) {
    int max_layer_size = 0;
    max_layer_size = max(Embedding_numel(self->wte), max_layer_size);
    max_layer_size = max(Linear_weight_numel(self->fc_1), max_layer_size);
    max_layer_size = max(Linear_weight_numel(self->fc_2), max_layer_size);
    return malloc(sizeof(float) * 2 * max_layer_size);  // Account for gradients.
}


// TODO(eugen): Consider sharding the bias as well, but usually not large enough to matter.
void Model_shard_fsdp(Model* self, int pg_rank, int pg_size) {
    // Shard wte.
    int wte_shard_size = Embedding_numel(self->wte) / pg_size;
    float* wte_shard = malloc(sizeof(float) * wte_shard_size);
    float* wte_d_shard = calloc(sizeof(float), wte_shard_size);
    memcpy(wte_shard, self->wte->embedding + (pg_rank * wte_shard_size), sizeof(float) * wte_shard_size);
    free(self->wte->embedding); self->wte->embedding = wte_shard;
    free(self->wte->d_embedding); self->wte->d_embedding = wte_d_shard;
    self->wte->vocab_size = self->wte->vocab_size / pg_size;

    // Shard fc_1.
    int fc_1_shard_size = Linear_weight_numel(self->fc_1) / pg_size;
    float* fc_1_shard = malloc(sizeof(float) * fc_1_shard_size);
    float* fc_1_d_shard = calloc(sizeof(float), fc_1_shard_size);
    memcpy(fc_1_shard, self->fc_1->weight + (pg_rank * fc_1_shard_size), sizeof(float) * fc_1_shard_size);
    free(self->fc_1->weight); self->fc_1->weight = fc_1_shard;
    free(self->fc_1->d_weight); self->fc_1->d_weight = fc_1_d_shard;
    self->fc_1->in_features = self->fc_1->in_features / pg_size;

    // Shard fc_2.
    int fc_2_shard_size = Linear_weight_numel(self->fc_2) / pg_size;
    float* fc_2_shard = malloc(sizeof(float) * fc_2_shard_size);
    float* fc_2_d_shard = calloc(sizeof(float), fc_2_shard_size);
    memcpy(fc_2_shard, self->fc_2->weight + (pg_rank * fc_2_shard_size), sizeof(float) * fc_2_shard_size);
    free(self->fc_2->weight); self->fc_2->weight = fc_2_shard;
    free(self->fc_2->d_weight); self->fc_2->d_weight = fc_2_d_shard;
    self->fc_2->in_features = self->fc_2->in_features / pg_size;
}


void Embedding_forward_fsdp(
    Embedding* self, int* idxs, Activation* output, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    int shard_size = Embedding_numel(self);
    allgather(self->embedding, shard_size, flat_buffer, pg_comm);
    float* shard = self->embedding;
    int shard_vocab_size = self->vocab_size;
    self->embedding = flat_buffer;
    self->vocab_size = shard_vocab_size * pg_size;
    Embedding_forward(self, idxs, output);
    self->embedding = shard;
    self->vocab_size = shard_vocab_size; 
}


void Embedding_backward_fsdp(
    Embedding* self, int* idxs, Activation* output, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    int shard_size = Embedding_numel(self);
    int full_size = shard_size * pg_size;
    memset(flat_buffer, 0, sizeof(float) * 2 * full_size);
    allgather(self->embedding, shard_size, flat_buffer, pg_comm);
    float* shard = self->embedding;
    float* d_shard = self->d_embedding;
    int shard_vocab_size = self->vocab_size;
    self->embedding = flat_buffer;
    self->d_embedding = flat_buffer + full_size;
    self->vocab_size = shard_vocab_size * pg_size;
    Embedding_backward(self, idxs, output);
    reducescatter_mean(flat_buffer + full_size, d_shard, shard_size, pg_comm, pg_size);
    self->embedding = shard;
    self->d_embedding = d_shard;
    self->vocab_size = shard_vocab_size; 
}


void Linear_forward_fsdp(
    Linear* self, Activation* input, Activation* output, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    int shard_size = Linear_weight_numel(self);
    allgather(self->weight, shard_size, flat_buffer, pg_comm);
    float* shard = self->weight;
    int shard_in_features = self->in_features;
    self->weight = flat_buffer;
    self->in_features = shard_in_features * pg_size;
    Linear_forward(self, input, output);
    self->weight = shard;
    self->in_features = shard_in_features;
}


void Linear_backward_fsdp(
    Linear* self, Activation* input, Activation* output, float* flat_buffer, MPI_Comm pg_comm, int pg_size
) {
    int shard_size = Linear_weight_numel(self);
    int full_size = shard_size * pg_size;
    memset(flat_buffer, 0, sizeof(float) * 2 * full_size);
    allgather(self->weight, shard_size, flat_buffer, pg_comm);
    float* shard = self->weight;
    float* d_shard = self->d_weight;
    int shard_in_features = self->in_features;
    self->weight = flat_buffer;
    self->d_weight = flat_buffer + full_size;
    self->in_features = shard_in_features * pg_size;
    Linear_backward(self, input, output);
    reducescatter_mean(flat_buffer + full_size, d_shard, shard_size, pg_comm, pg_size);
    self->weight = shard;
    self->d_weight = d_shard;
    self->in_features = shard_in_features;
}


// ========== Pipeline parallelism utils ==========

void Model_shard_pp(Model* self, int pg_rank) {
    if (pg_rank == 0) {
        Linear_destroy(self->fc_1); self->fc_1 = NULL;
        Linear_destroy(self->fc_2); self->fc_2 = NULL;
        Activation_destory(self->fc_1_out); self->fc_1_out = NULL;
        Activation_destory(self->relu_out); self->relu_out = NULL;
        Activation_destory(self->fc_2_out); self->fc_2_out = NULL;
        Activation_destory(self->softmax_out); self->softmax_out = NULL;
    } else if (pg_rank == 1) {
        Embedding_destory(self->wte); self->wte = NULL;
        Linear_destroy(self->fc_2); self->fc_2 = NULL;
        Activation_destory(self->fc_2_out); self->fc_2_out = NULL;
        Activation_destory(self->softmax_out); self->softmax_out = NULL;
    } else if (pg_rank == 2) {
        Embedding_destory(self->wte); self->wte = NULL;
        Linear_destroy(self->fc_1); self->fc_1 = NULL;
        Activation_destory(self->wte_out); self->wte_out = NULL; self->wte_out_flat = NULL;
        Activation_destory(self->fc_1_out); self->fc_1_out = NULL;
    } else {
        printf("Unknown rank: %d\n", pg_rank);
        MPI_Finalize();
        exit(1);
    }
}


void Model_zerograd_pp(Model* self, int pg_rank) {
    if (pg_rank == 2) {
        memset(self->fc_2->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_2));
        memset(self->fc_2->d_bias, 0, sizeof(float) * self->fc_2->out_features);
        memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
        memset(self->fc_2_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_2_out));
        memset(self->softmax_out->d_value, 0, sizeof(float) * Activation_numel(self->softmax_out));
    } else if (pg_rank == 1) {
        memset(self->fc_1->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_1));
        memset(self->fc_1->d_bias, 0, sizeof(float) * self->fc_1->out_features);
        memset(self->wte_out_flat->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
        memset(self->fc_1_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_1_out));
        memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
    } else if (pg_rank == 0) {
        memset(self->wte->d_embedding, 0, sizeof(float) * Embedding_numel(self->wte));
        memset(self->wte_out->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
    } else {
        printf("Unknown rank: %d\n", pg_rank);
        MPI_Finalize();
        exit(1);
    }
}


void Model_step_pp(Model* self, float lr, int pg_rank) {
    if (pg_rank == 0) {
        sgd_step(self->wte->embedding, self->wte->d_embedding, Embedding_numel(self->wte), lr);
    } else if (pg_rank == 1) {
        sgd_step(self->fc_1->weight, self->fc_1->d_weight, Linear_weight_numel(self->fc_1), lr);
        sgd_step(self->fc_1->bias, self->fc_1->d_bias, self->fc_1->out_features, lr);
    } else if (pg_rank == 2) {
        sgd_step(self->fc_2->weight, self->fc_2->d_weight, Linear_weight_numel(self->fc_2), lr);
        sgd_step(self->fc_2->bias, self->fc_2->d_bias, self->fc_2->out_features, lr);
    } else {
        printf("Unknown rank: %d\n", pg_rank);
        MPI_Finalize();
        exit(1);
    }
}