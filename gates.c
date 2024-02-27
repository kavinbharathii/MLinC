#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

#define trainCount (sizeof(train) / sizeof(train[0]))

float randomFloat(void) {
    return ((float) rand() / (float) RAND_MAX);
}

float costFunction(float w1, float w2) {

    float result = 0.0f;

    for (int i = 0; i < trainCount; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float yhat = (x1 * w1) + (x2 * w2);
        float error = yhat - train[i][2];
        result += error*error;
    }

    // root mean squared error
    result = sqrt(result / trainCount);
    return result;
}

float derivativeOf(float (*cost)(float, float), float w1, float w2, bool token) {
    float epsilon = 1e-3;
    float costwb = cost(w1, w2);

    if (!token) {
        return (cost(w1 + epsilon, w2) - costwb) / epsilon;
    } else {
        return (cost(w1, w2 + epsilon) - costwb) / epsilon;
    }
    
}

void modelPerformance(float w1, float w2, char token[]) {
    printf("%s: w1: %f,  w2: %f\n", token, w1, w2);

    for (int i = 0; i < trainCount; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float yhat = (x1 * w1) + (x2 * w2);
        
        printf("Actual: %d, Predicted: %d\n", (int)round(train[i][2]), (int)round(yhat));
    }

    printf("\n");
}

void main() {
    srand(100);
    float w1 = randomFloat()*10.0f;
    float w2 = randomFloat()*10.0f;

    float dw1, dw2;
    float cost;
    float learningRate = 1e-3;
    int steps = 0;

    modelPerformance(w1, w2, "Before");
    while (steps < 1000) {
        dw1 = derivativeOf(costFunction, w1, w2, false);
        dw2 = derivativeOf(costFunction, w1, w2, true);
        w1 -= (learningRate * dw1);
        w2 -= (learningRate * dw2);
        steps += 1;

    }
    modelPerformance(w1, w2, "After");
}
