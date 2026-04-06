#ifndef MODEL_C
#define MODEL_C

#include <string.h>
#include <stdbool.h>
#include "ops.c"


void sgd_step(float* param, float* d_param, int size, float lr) {
    for (int i = 0; i < size; i++) {
        param[i] += -lr * d_param[i];
    }
}


typedef struct {
    Embedding* wte;
    Linear* fc_1;
    Linear* fc_2;

    Activation* wte_out;
    Activation* wte_out_flat;
    Activation* fc_1_out;
    Activation* relu_out;
    Activation* fc_2_out;
    Activation* softmax_out;
} Model;


Model* Model_create(int batch_size, int seq_len, int vocab_size, int emb_size, int hidden_size) {
    // Create parameters.
    Embedding* wte = Embedding_create(vocab_size, emb_size);
    Linear* fc_1 = Linear_create(seq_len * emb_size, hidden_size);
    Linear* fc_2 = Linear_create(hidden_size, vocab_size);
    
    // Create activations.
    Activation* wte_out = Activation_create(batch_size * seq_len, emb_size);
    Activation* wte_out_flat = Activation_view(wte_out, batch_size, seq_len * emb_size);
    Activation* fc_1_out = Activation_create(batch_size, hidden_size);
    Activation* relu_out = Activation_create(batch_size, hidden_size);
    Activation* fc_2_out = Activation_create(batch_size, vocab_size);
    Activation* softmax_out = Activation_create(batch_size, vocab_size);

    Model* self = malloc(sizeof(Model));
    self->wte = wte;
    self->fc_1 = fc_1;
    self->fc_2 = fc_2;
    self->wte_out = wte_out;
    self->wte_out_flat = wte_out_flat;
    self->fc_1_out = fc_1_out;
    self->relu_out = relu_out;
    self->fc_2_out = fc_2_out;
    self->softmax_out = softmax_out;
    return self;
}


float Model_forward(Model* self, int* Xs, int* Ys) {
    Embedding_forward(self->wte, Xs, self->wte_out);
    Linear_forward(self->fc_1, self->wte_out_flat, self->fc_1_out);
    relu(self->fc_1_out, self->relu_out);
    Linear_forward(self->fc_2, self->relu_out, self->fc_2_out);
    softmax(self->fc_2_out, self->softmax_out);
    return cross_entropy_loss(self->softmax_out, Ys);
}


void Model_zerograd(Model* self) {
    memset(self->wte->d_embedding, 0, sizeof(float) * Embedding_numel(self->wte));
    memset(self->fc_1->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_1));
    memset(self->fc_1->d_bias, 0, sizeof(float) * self->fc_1->out_features);
    memset(self->fc_2->d_weight, 0, sizeof(float) * Linear_weight_numel(self->fc_2));
    memset(self->fc_2->d_bias, 0, sizeof(float) * self->fc_2->out_features);
    memset(self->wte_out->d_value, 0, sizeof(float) * Activation_numel(self->wte_out));
    memset(self->fc_1_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_1_out));
    memset(self->relu_out->d_value, 0, sizeof(float) * Activation_numel(self->relu_out));
    memset(self->fc_2_out->d_value, 0, sizeof(float) * Activation_numel(self->fc_2_out));
    memset(self->softmax_out->d_value, 0, sizeof(float) * Activation_numel(self->softmax_out));
}


void Model_backward(Model* self, int* Xs, int* Ys) {
        Model_zerograd(self);
        cross_entropy_softmax_backward(self->fc_2_out, self->softmax_out, Ys);
        Linear_backward(self->fc_2, self->relu_out, self->fc_2_out);
        relu_backward(self->fc_1_out, self->relu_out);
        Linear_backward(self->fc_1, self->wte_out_flat, self->fc_1_out);
        Embedding_backward(self->wte, Xs, self->wte_out);
}


void Model_step(Model* self, float lr) {
    sgd_step(self->wte->embedding, self->wte->d_embedding, Embedding_numel(self->wte), lr);
    sgd_step(self->fc_1->weight, self->fc_1->d_weight, Linear_weight_numel(self->fc_1), lr);
    sgd_step(self->fc_1->bias, self->fc_1->d_bias, self->fc_1->out_features, lr);
    sgd_step(self->fc_2->weight, self->fc_2->d_weight, Linear_weight_numel(self->fc_2), lr);
    sgd_step(self->fc_2->bias, self->fc_2->d_bias, self->fc_2->out_features, lr);
}


int Model_sample_token(Model* self) {
    int tok = -1;
    float u = uniform();
    float curr_density = 0;
    for (int i = 0; i < self->softmax_out->features; i++) {
        curr_density += self->softmax_out->value[i];
        if (curr_density >= u) {
            tok = i;
            break;
        }
    }
    return tok;
}


bool Model_sample_update_input(int* Xs, int* Ys, int tok, int seq_len) {
    // If this is a <BOS> token, we're done so just return.
    if (tok == 0) {
        return true;
    }
    // Otherwise, shift Xs one to the left, add the new token, and keep sampling.
    for (int i = 0; i < seq_len - 1; i++) {
        Xs[i] = Xs[i + 1];
    }
    Xs[seq_len - 1] = tok;
    return false;
}


// TODO(eugen): Add support for sampling a full batch at a time.
void Model_sample(Model* self, int* Xs, int* Ys, int seq_len) {
    bool done = false;
    while (!done) {
        Model_forward(self, Xs, Ys);
        int tok = Model_sample_token(self);
        done = Model_sample_update_input(Xs, Ys, tok, seq_len); 
    }
}


#endif
