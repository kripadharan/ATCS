//Author: Kripa Dharan

#include "stdio.h"
#include "string.h"
#include "stdlib.h"

typedef enum {true, false} bool;

const int MAX_DATA = 50000;
const int MAX_COMMANDS = 10000;
int data[MAX_DATA];
int location = 0;
char commands[MAX_COMMANDS];
int com = 0;
void execute(char ch);
void paren();

void initialize_data(int data[]){
	for(int i = 0; i < MAX_DATA; ++i){
		data[i] = 0;
	}
}

int populate_commands(FILE* fp){
	int count_paren = 0;
	int count = 0;
	char ch;
	while((ch = fgetc(fp)) != EOF){
		if(ch == '>' || ch == '<' || ch == '+' || ch == '-' || ch == '(' || ch == ')' || ch == '*'){
			commands[count] = ch;
			count += 1;
			if(ch == '('){
				count_paren += 1;
			}
			if(ch == ')'){
				count_paren -= 1;
			}
			if(count_paren < 0){
				printf("Invalid set of commands - Invalid combination of parentheses");
				exit(1);
			}
		}
		
	}
	if(count_paren > 0){
		printf("Invalid set of commands - Invalid combination of parentheses");
		exit(1);
	}
	return count;
}

FILE* open_file(char* file_name){
	FILE* fp = fopen(file_name, "r");
	if (fp == NULL) {
		printf("Uhoh, couldn’t open file!");
		exit(1);
	}
	return fp;
}

void execute(char ch)
{
	if(ch == '>'){
		if(location == MAX_DATA - 1){
			printf("Invalid instruction: Pointress cannot move right if she is already pointing to the right-most villager.\n");
			exit(1);
		}
		location += 1;
	}
	else if(ch == '<'){
		if(location == 0){
			printf("%d\n", com);
			printf("Invalid instruction: Pointress cannot move left if she is already pointing to the left-most villager.\n");
			exit(1);
		}
		location -= 1;
	}
	else if(ch == '+'){
		if(data[location] == 255){
			data[location] = 0;
		}
		else{
			data[location] += 1;
		}
	}
	else if(ch == '-'){
		if(data[location] == 0){
			data[location] = 255;
		}
		else{
			data[location] -= 1;
		}
	}
	else if(ch == '*'){
		printf("%c", (char)data[location]);
	}
	else if(ch == '('){
		paren();
	}
		com += 1;
}

void paren(){
	bool is_zero;
	int start = com;
	if(data[location] == 0){
		is_zero = true;
	}
	else{
		is_zero = false;
	}
	char ch;
	if(is_zero == false){
		com += 1;
		while((ch = commands[com]) != ')'){
			execute(ch);
		}
		if(data[location] == 0){
			return;
		}
		else{
			com = start-1;
			return;
		}
	}
	else{
		bool done = false;
		int count = 0;
		while(done == false){
			char current = commands[com];
			if(current == ')'){
				count -= 1;
			}
			if(current == '('){
				count += 1;
			}
			if(current == ')' && count == 0){
				return;
			}
			com += 1;
		}
	}
	
}


int main(int argc, char** argv){
	initialize_data(data);
	FILE* fp = open_file("/Users/kripadharan/Documents/ATCS/ATCS_y4_-kripadharan/program.bf");
	int num_commands = populate_commands(fp);
	char ch;
	while(com < num_commands){
		ch = commands[com];
		execute(ch);
	}
	printf("\n");
}



