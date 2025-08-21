
#include "opengl.h"
#include "simulationstate.h"

int main()
{
    bool fullScreen = true;
    InteropOpenGL OpenGL(1920, 1080, "Cuda OpenGL Interop", fullScreen);
    OpenGL.disableVSYNC();

    SimulationState simState;
    initSimulation(OpenGL.getScreenDim(), simState);

    while (OpenGL.isAlive())
    {
        OpenGL.processUserInput();
        OpenGL.executeKernels(simState);
        OpenGL.renderFullScreenQuad();
        OpenGL.renderImGui(simState);
        OpenGL.swapBuffers();
    }

    simState.balls.free();
    OpenGL.free();
    return 0;
}

