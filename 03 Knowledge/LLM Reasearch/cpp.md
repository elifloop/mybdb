## 1. History

- programming happens within a context (computing machines available at the time)
- started in 1800 by jacquard's loom where info was stored in card
- programming lang is a formal language (syntax) which is recognizable by a computer & has unambigious intent (semantics) to compute
- language hiereachy
   - machine language - processor specific like x86 assembly differs from ARM or MIPS assembly;
   - assembly language - an assembler translates into machine code
   - high level languages - High-level code is translated to machine-executable form in two common ways;
                           1. Compiler - Translates the whole program before execution, often producing an executable file	C, C++, C#, Rust, Go
                           2. Interpreter - Interpreter	Translates and executes code line by line while the program runs	Python, JavaScript, Ruby, PHP
                           3. Bytecode + virtual machine -	Compiles to an intermediate form, then runs it on a virtual machine	Java, which compiles to bytecode for the JVM
     

## 2. Hello World code

```
#include <iostream>

using namespace std;

int main()
{
   cout << "Hello World!" << endl;
   cin.get();

   return 0;
}
```

- `#include <iostream>`
⚒️ The include command is a specific preprocessor command that effectively copies and pastes the entire text of the file specified between the angle brackets into the source code. In this case the file is "iostream" which is a standard file that should come with the C++ compiler.
⚒️ allows a programmer to "include" this functionality in the program without having to literally cut and paste it into the source code every time
⚒️ The iostream file is part of the C++ standard library, which provides a set of useful and commonly used functionality provided with the compiler. 
⚒️ The "include" mechanism, however, can be used both for standard code provided by the compiler and for reusable files created by the programmer.

- `using name space std;`
  ⚒️ A namespace is essentially a prefix that is applied to all the names in a certain set. One way to think about namespaces is that they are like toolboxes with different useful tools
  ⚒️ The using command tells the compiler to allow all the names in the "std" namespace to be usable without their prefix
  ⚒️ The iostream file defines three names used in this program - cout, cin, and endl - which are all defined in the std namespace.
  ⚒️ Without using the std namespace, the names would have to include the prefix and be written as std::cout, std::cin, and std::endl
  ⚒️ consider it a small price to pay for avoiding all the tedious work of constantly retyping  std::free in front of things in your program
  
- `int main()`
  ⚒️ starting point of all C++ programs is the main function.  & this function is called by the operating system when your program is executed by the computer

- `cin, cout`
  ⚒️ name cout is short for "character output" and cin for "character input"
  ⚒️ In a typical C++ program, most function calls are of the form object.function_name(argument1, argument2) ; here cin is the object and get is a function and it doesnt have args
  ⚒️ symbols such as << can also behave as functions, as illustrated by the use of cout above. This capability is called operator overloading which will be discussed later on
  
- `other keywords`
  ⚒️ `{}` - A block of code is defined with the { } tokens. { signifies the start of a block of code and } signifies the end. It has other use as well
  ⚒️ `;' - all sentences must end with semicolons
  ⚒️ `return` - keyword tells the program to return a value to the function that called this function and then to continue execution in the calling function from the point at which this function was called. The type of the value returned by a function must match the type specified in the declaration of the function. **Executing the return keyword in the main function of a program returns a value and the execution control to the operating system component that launched this program, in effect, terminating the execution of this program.**
  
## 3. Code compilation
- The compiler translates the textual representation of the program into a form that a computer can execute more efficiently.
- A high-level language like C++ is actually a sort of 'compromise' language between the native language of the CPU (generally referred to as machine language) and the native language of the programmer (say English).
- C++ is a compiled lang ; on the contrary Java, py are interpreted lang; an interpreter is like a compiler that converts the program into machine language at the time it is run on the computer rather than in advance as is done with a compiled language
- For MacOS X, there is Xcode which uses gcc for compiling C++. For Windows, there is Dev C++ which also uses gcc for compiling C++, Microsoft Visual C++ (and its free Express version), TCLITE, and ports of the GNU Compiler Collection distributed within Cygwin and MinGW.
- Each compiler is invoked in a specific way. For example, if you wish to use GCC, type the following into a terminal: c++ main.cpp -o example -> main.cpp will be compiled and will generate an executable file called example
- to invoke a compiled file you use ./example in this case
- [using clang compiler] - clang++ -std=c++20 -Wall -Wextra -g 01_basics/hello.cpp -o hello

## 4. Exercise examples

#### e1 - Print 2 lines of code 
#include<iostream>
using name space std;
intmain()
{
 cout<<"Hello, World!"<<endl;
 cout<<"Live long and prosper."<<endl;
 cin.get();
 return 0;
} 
#### e2 - Print the same thing with out using name space...
#include<iostream>
int main()
{
 std::cout<<"Hello, World!"<<std::endl;
 std::cout<<"Live long and prosper."<<std::endl;
 std::cin.get();
 return 0;
} 

## 5. keywords, vars, constants in C

**32 reserved keywords are there** 

auto	double	int	struct
break	else	long	switch
case	enum	register	typedef
char	extern	return	union
const	float	short	unsigned
continue	for	signed	void
default	goto	sizeof	volatile
do	if	static	while

**How to assign vars **
char abc = 'X';
int marks =77;
float amount = 45.23;

## 6. dtypes, keywords, vars, constants in C++

dtypes
- int - 16 bits
- float - 32 bits (7 decimal places behind a comma)
- bool - 1 bit
- char - 8 bits
- string - unknown
- double - doubles the value of a float - 64 bits (14 decimal places behind a comma) 

keywords
- signed - gives a variable possible to carry -ve  or +ve val
- unsigned - forces value to be positive therefore it can be twice as large
- long - doubles the val of the varible ; so a long int would be 32 bits etc...
- long long - doubles the val and then doubles it again

vars
- type name; type name = value; modifier type name;
- example- int myInt; int myintt = 20; unsigned int myinttt;

