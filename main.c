#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
    {10, 20},
    {32, 64},
    {99, 198},
    {2143, 4286},
    {11, 22},
};

float test[][2] = {
    {12, 24},
    {32, 64},
    {1080, 2160},
    {55, 110},
    {1423, 2846},
};

#define trainCount (sizeof(train) / sizeof(train[0]))
#define testCount (sizeof(test) / sizeof(test[0]))

float randomFloat(void) {
    return ((float) rand() / (float) RAND_MAX);
}

float sigmoidf(float x) {
    return 1 / (1 + exp(-x));
}

float costFunction(float w) {

    float result = 0.0f;

    for (int i = 0; i < trainCount; i++) {
        float x = train[i][0];
        float yhat = (x * w);
        float error = yhat - train[i][1];
        result += error*error;
    }

    // root mean squared error
    result = sqrt(result / trainCount);
    return result;
}

float derivativeOf(float (*cost)(float), float w) {
    float epsilon = 1e-3;
    return (cost(w + epsilon) - cost(w)) / epsilon;
}

void modelPerformance(float w) {
    for (int i = 0; i < testCount; i++) {
        float x = test[i][0];
        float yhat = (x * w);
        
        printf("Actual: %4d, Predicted: %4d\n", (int)test[i][1], (int)yhat);
    }
    printf("\n");
}

void main() {
    srand(100);
    float w = randomFloat();

    float dw;
    float cost;
    float learningRate = 1e-4;
    int steps = 0;

    while (steps < 5000) {
        dw = derivativeOf(costFunction, w);
        w -= (learningRate * dw);
        steps += 1;

        // adjusting the learning rate as the training progresses
        if ((steps % 1000) == 0) {
            learningRate /= 10;
        }
    }
    printf("%f\n", w);
    modelPerformance(w);
}
