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

float costFunction(float w1, float w2, float b) {

    float result = 0.0f;

    for (int i = 0; i < trainCount; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float yhat = (x1 * w1) + (x2 * w2) + b;
        float error = yhat - train[i][2];
        result += error*error;
    }

    // root mean squared error
    result = sqrt(result / trainCount);
    return result;
}

void modelPerformance(float w1, float w2, float b, char token[]) {
    printf("%s: w1: %f,  w2: %f, b: %f\n", token, w1, w2, b);

    for (int i = 0; i < trainCount; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float yhat = (x1 * w1) + (x2 * w2) + b;
        
        printf("Actual: %d, Predicted: %d\n", (int)round(train[i][2]), (int)round(yhat));
    }

    printf("\n");
}

void main() {
    srand(100);
    float w1 = randomFloat();
    float w2 = randomFloat();
    float b = randomFloat();
    float epsilon = 1e-3;

    float dw1, dw2, db;
    float cost;
    float learningRate = 1e-3;
    int steps = 0;

    modelPerformance(w1, w2, b, "Before");
    while (steps < 1000) {
        float costwb = costFunction(w1, w2, b);
        dw1 = (costFunction(w1 + epsilon, w2, b) - costwb) / epsilon;
        dw2 = (costFunction(w1, w2 + epsilon, b) - costwb) / epsilon;
        db = (costFunction(w1, w2, b + epsilon) - costwb) / epsilon;
        w1 -= (learningRate * dw1);
        w2 -= (learningRate * dw2);
        b -= (learningRate * db);
        steps += 1;

    }
    modelPerformance(w1, w2, b, "After");
}