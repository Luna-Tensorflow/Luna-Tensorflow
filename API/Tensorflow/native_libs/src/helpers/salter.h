//
// Created by mateusz on 05.12.18.
//

#ifndef TFL_SALTER_HPP
#define TFL_SALTER_HPP

#include <random>
#include <climits>
#include <map>

class salter
{
private:
	std::mt19937 engine;
	std::uniform_int_distribution<> distribution;

public:
	salter(){
		engine = std::mt19937(std::random_device{}());
		distribution = std::uniform_int_distribution<>(0, INT_MAX);
	}

	size_t next_salt()
	{
		return static_cast<size_t>(distribution(engine));
	}
};

extern salter const_salter;


#endif //TFL_SALTER_HPP
