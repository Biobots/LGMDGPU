#ifndef APP
#define APP

#include <kompute/Kompute.hpp>
#include <vector>
#include <screen.hpp>

class Application
{
private:
	kp::Manager manager;
	Screen screen;
	std::shared_ptr<kp::Tensor> tensor;
public:
	Application()
	{
		manager = kp::Manager();
		
	}
	void initializeScreen(int width, int height, float hexRad)
	{
		screen = Screen(&manager, width, height, hexRad);
	}
	~Application()
	{

	}
};

#endif // !APP