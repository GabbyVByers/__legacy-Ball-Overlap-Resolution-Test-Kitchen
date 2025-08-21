#pragma once

#include "sharedarray.h"
#include "random.h"
#include <tuple>

struct Ball
{
    Vec2f currPos;
    Vec2f prevPos;
    Vec2f acceleration;

    Vec2f new_currPos; // these are for resolving ball-to-ball collisions, not performing verlet integration
    Vec2f new_prevPos;
    bool wasCrowded = false;

    float radius = 0.0f;
    float mass = 1.0f;
    uchar4 color;
};

struct SimulationState
{
    // global
    int screenWidth = -1;
    int screenHeight = -1;
    float max_u = 0.0f;
    uchar4* pixels = nullptr;
    SharedArray<Ball> balls;
    float dt = 0.0001f;
    float wallCollisionDampening = 0.99f;
    float ballCollisionDampening = 0.99f;
    float gravity = 200.0f;
};

inline void initSimulation(std::tuple<int, int> screenDim, SimulationState& simState)
{
    simState.screenWidth  = std::get<0>(screenDim);
    simState.screenHeight = std::get<1>(screenDim);
    simState.max_u = (simState.screenWidth / (float)simState.screenHeight);;

    int numBalls = 600;
    for (int i = 0; i < numBalls; i++)
    {
        Ball ball;
        ball.radius = 0.03f;
        ball.mass = 3.14f * ball.radius * ball.radius;
        ball.currPos.x = randomFloat(-simState.max_u + ball.radius, simState.max_u - ball.radius);
        ball.currPos.y = randomFloat(-1.0f + ball.radius, 1.0f - ball.radius);
        Vec2f initVelocity = randomVec2f(-1.0f, 1.0f);
        normalize(initVelocity);
        ball.prevPos = ball.currPos + (initVelocity / 1000.0f);
        ball.color = randomColor();
        simState.balls.add(ball);
    }
    simState.balls.updateHostToDevice();
}

