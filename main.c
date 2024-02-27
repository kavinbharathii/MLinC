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

float costFunction(float w, float b) {

    float result = 0.0f;

    for (int i = 0; i < trainCount; i++) {
        float x = train[i][0];
        float yhat = (x * w) + b;
        float error = yhat - train[i][1];
        result += error*error;
    }

    // root mean squared error
    result = sqrt(result / trainCount);
    return result;
}

float derivativeOf(float (*cost)(float, float), float w, float b, bool bias) {
    float epsilon = 1e-3;
    float costwb = cost(w, b);

    if (!bias) {
        return (cost(w + epsilon, b) - costwb) / epsilon;
    } else {
        return (cost(w, b + epsilon) - costwb) / epsilon;
    }
    
}

void modelPerformance(float w, float b) {
    for (int i = 0; i < testCount; i++) {
        float x = test[i][0];
        float yhat = (x * w) + b;
        
        printf("Actual: %d, Predicted: %d\n", (int)test[i][1], (int)yhat);
    }
}

void main() {
    srand(100);
    float w = randomFloat()*10.0f;
    float b = randomFloat()*5.0f;

    float dw, db;
    float cost;
    float learningRate = 1e-3;
    int steps = 0;

    while (steps < 500) {
        printf("weight: %f,  bias: %f\n", w, b);
        dw = derivativeOf(costFunction, w, b, false);
        db = derivativeOf(costFunction, w, b, true);
        w -= (learningRate * dw);
        b -= (learningRate * db);
        steps += 1;

    }
    printf("%f\n", w);
    printf("%f\n", b);
    modelPerformance(w, b);
}
