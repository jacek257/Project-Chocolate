#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>

int main()
{
  long long test = 2*(
    29.0/(1 + exp(-.6118 *(0.0 - 42.115)))+3.0 - 4.0
  );
  std::cout<<test<<std::endl;
}
