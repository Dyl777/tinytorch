#include <math.h>
#include <stdlib.h>


typedef struct {
    int batch_size;
    int features;
    float* value;
    float* d_value;
} Activation;


Activation* Activation_create(int batch_size, int features) {
    Activation* self = malloc(sizeof(Activation));
    self->batch_size = batch_size;
    self->features = features;
    self->value = calloc(sizeof(float), batch_size * features);
    self->d_value = calloc(sizeof(float), batch_size * features);
    return self;
}


void Activation_destory(Activation* self) {
    free(self->value);
    free(self->d_value);
    free(self);
}


Activation* Activation_view(Activation* other, int batch_size, int features) {
    Activation* self = malloc(sizeof(Activation));
    self->batch_size = batch_size;
    self->features = features;
    self->value = other->value;
    self->d_value = other->d_value;
    return self;
}


int Activation_numel(Activation* self) {
    return self->batch_size * self->features;
}


typedef struct {
    int in_features;
    int out_features;
    float* weight;
    float* d_weight;
    float* bias;
    float* d_bias;
} Linear;


float uniform() {
    return (float)rand() / RAND_MAX;	
}


float he_init(float k) {
    return 2 * k * uniform() - k;
}


Linear* Linear_create(int in_features, int out_features) {
    float* weight = malloc(sizeof(float) * in_features * out_features);
    float* d_weight = malloc(sizeof(float) * in_features * out_features);
    float* bias = malloc(sizeof(float) * out_features);
    float* d_bias = malloc(sizeof(float) * out_features);

    // Initalize weights, biases, and gradients.
    float k = 1.0f / (float)in_features;
    for (int i = 0; i < in_features * out_features; i++) {
        weight[i] = he_init(k);
        d_weight[i] = 0.0f;
    }
    for (int i = 0; i < out_features; i ++){
        bias[i] = he_init(k);
        d_bias[i] = 0.0f;
    }
 
    Linear* self = malloc(sizeof(Linear));
    self->in_features = in_features;
    self->out_features = out_features;
    self->weight = weight;
    self->d_weight = d_weight;
    self->bias = bias;
    self->d_bias = d_bias;

   return self;
}


void Linear_destroy(Linear* self) {
    free(self->weight);
    free(self->d_weight);
    free(self->bias);
    free(self->d_bias);
    free(self);
}


int Linear_weight_numel(Linear* self) {
    return self->in_features * self->out_features;
}


void Linear_forward(Linear* self, Activation* input, Activation* output) {
    // Y = X @ W + B
    for (int b = 0; b < input->batch_size; b++) {
        for (int o = 0; o < self->out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < self->in_features; i++) {
                int input_idx = b * self->in_features + i;
                int weight_idx = i * self->out_features + o;
                sum += input->value[input_idx] * self->weight[weight_idx];
            }
            output->value[b * self->out_features + o] = sum + self->bias[o];
        }
    }
}


void Linear_backward(Linear* self, Activation* input, Activation* output) {
    // dL/dX = dL/dY @ W.T 
    for (int b = 0; b < input->batch_size; b++) {
        for (int i = 0; i < self->in_features; i++) {
            float sum = 0.0f;
            for (int o = 0; o < self->out_features; o++) {
                int d_output_idx = b * self->out_features + o;
                int weight_idx = i * self->out_features + o;
                sum += output->d_value[d_output_idx] * self->weight[weight_idx];
            }
            input->d_value[b * self->in_features + i] += sum;
        }
    }

    // dL/dW = X.T @ dL/dY
    for (int i = 0; i < self->in_features; i++) {
        for (int o = 0; o < self->out_features; o++) {
            float sum = 0.0f;
            for (int b = 0; b < input->batch_size; b++) {
                int input_idx = b * self->in_features + i;
                int d_output_idx = b * self->out_features + o;
                sum += input->value[input_idx] * output->d_value[d_output_idx];
            }
            self->d_weight[i * self->out_features + o] += sum;
        }
    }

    // TODO(eugen): This might be more efficient to above with the dL/dW calculation
    // since the memory is already loaded.
    // dL/db = 1 @ dL/dY
    for (int o = 0; o < self->out_features; o++) {
        float sum = 0.0f;
        for (int b = 0; b < input->batch_size; b++) {
            sum += output->d_value[b * self->out_features + o];
        }
        self->d_bias[o] += sum;
    }
}


typedef struct {
    int vocab_size;
    int emb_size;
    float* embedding;
    float* d_embedding;
} Embedding;


Embedding* Embedding_create(int vocab_size, int emb_size) {
    float* embedding = malloc(sizeof(float) * vocab_size * emb_size);
    float* d_embedding = malloc(sizeof(float) * vocab_size * emb_size);
    float k = 1.0f / (float)emb_size;
    for (int i = 0; i < vocab_size * emb_size; i++) {
        embedding[i] = he_init(k);
        d_embedding[i] = 0.0f;
    }
 
    Embedding* self = malloc(sizeof(Embedding));
    self->vocab_size = vocab_size;
    self->emb_size = emb_size;
    self->embedding = embedding;
    self->d_embedding = d_embedding;
    return self;
}


void Embedding_destory(Embedding* self) {
    free(self->embedding);
    free(self->d_embedding);
    free(self);
}


int Embedding_numel(Embedding* self) {
    return self->vocab_size * self->emb_size;
}


void Embedding_forward(Embedding* self, int* idxs, Activation* output) {
    for (int b = 0; b < output->batch_size; b++) {
        for (int i = 0; i < output->features; i++) {
            int out_idx = b * output->features + i;
            int emb_idx = idxs[b] * output->features + i;
            output->value[out_idx] = self->embedding[emb_idx];
        }
    }
}


// TODO(eugen): Keeping a d_embedding is not actually necessary and wastes memory. 
// All we need to do is route the gradient of the input back into the embedding table.
void Embedding_backward(Embedding* self, int* idxs, Activation* output) {
    for (int b = 0; b < output->batch_size; b++) {
        for (int i = 0; i < output->features; i++) {
            int emb_idx = idxs[b] * output->features + i;
            int out_idx = b * output->features + i;
            self->d_embedding[emb_idx] += output->d_value[out_idx];
        }
    }
}


void relu(Activation* input, Activation* output) {
    for (int i = 0; i < Activation_numel(input); i++) {
        output->value[i] = input->value[i] > 0 ? input->value[i] : 0.0f;
    }
}


void relu_backward(Activation* input, Activation* output) {
    for (int i = 0; i < Activation_numel(input); i++) {
        input->d_value[i] += input->value[i] > 0 ? output->d_value[i] * 1.0f : 0.0f;
    }
}


void softmax(Activation* logits, Activation* probs) {
    for (int b = 0; b < logits->batch_size; b++) {
        // Find the max value of the row.
        float max_value = -INFINITY;
        float Z = 0;
        for (int v = 0; v < logits->features; v++) {
            int idx = b * logits->features + v;
            float prev_max = max_value;
            max_value = logits->value[idx] > max_value ? logits->value[idx] : max_value;
            Z *= exp(prev_max - max_value);
            Z += exp(logits->value[idx] - max_value);
        }
        // Compute stable softmax.
        for (int v = 0; v < logits->features; v++) {
            int idx = b * logits->features+ v;
            probs->value[idx] = exp(logits->value[idx] - max_value) / Z;
        }
    }
}


float cross_entropy_loss(Activation* probs, int* target) {
    float loss = 0.0f;
    for (int b = 0; b < probs->batch_size; b++) {
        int idx = b * probs->features + target[b];
        loss += log(probs->value[idx]);
    }
    return -(loss / probs->batch_size);
}


void cross_entropy_softmax_backward(Activation* logits, Activation* probs, int* target) {
    float d_loss = 1.0f / probs->batch_size;
    for (int b = 0; b < probs->batch_size; b++) {
        for (int v = 0; v < probs->features; v++) {
            int idx = b * probs->features+ v;
            int indicator = v == target[b] ? 1.0f : 0.0f;
            logits->d_value[idx] += d_loss * (probs->value[idx] - indicator);
        }
    }
}

