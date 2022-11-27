% -- Part 1 --

% 1.1 --- Multiples of Three
array[int] of int: multiples_of_three ::output = [ 3 * i | i in 1..10 ];

% 1.2 --- Even Multiples of Three
array[int] of int: even_multiples_of_three ::output = [ i | i in multiples_of_three where i mod 6 = 0];

% 1.3 --- Products
array[int] of int: products ::output = [ i * j | i in 1..10, j in 1..10 ];
include "alldifferent.mzn";

array[int] of int: products_div_3_not_2 ::output = [i * j | i in 1..10, j in 1..10 where i * j mod 3 =0 /\ i*j mod 2 != 0 /\ i < j];

% -- Part 2 --

% 2.1 --- Manhattan Distance

enum Point = P(1..5);
array[Point] of int: x_coord = [1,3,5,1,10];
array[Point] of int: y_coord = [4,3,8,1,7];

array[Point,Point] of int: manhattan_distance ::output = [ (i,j): abs(x_coord[i] - x_coord[j]) + abs(y_coord[i] - y_coord[j]) | i, j in Point];

% 2.2 --- Manhattan Distance with Dummy

enum Point0 = { Dummy } ++ PP(Point);
array[Point0,Point0] of int: manhattan_distance0 ::output = [ (i,j): 
            if i = Dummy \/ j = Dummy then 0 else manhattan_distance[to_enum(Point,i-1),to_enum(Point,j-1)] endif | i, j in Point0];

% 2.3 --- Euclidian Distance with Dummy

array[Point,Point] of float: euclidian_distance ::output = [ (i,j): sqrt((x_coord[i] - x_coord[j])^2 + (y_coord[i] - y_coord[j])^2) | i, j in Point ];

array[Point0,Point0] of float: euclidian_distance0 ::output = [ (i,j): 
            if i = Dummy \/ j = Dummy then 0 else euclidian_distance[to_enum(Point,i-1),to_enum(Point,j-1)] endif | i, j in Point0];

% -- Part 3 --

% 3.1  --- All Different

int: n = 5;
array[1..n] of var 1..n: x ::output;

% TODO: Enforce all_different any way you can think of
constraint forall(i,j in 1..n where i != j)(x[i] != x[j]);
constraint sum(i, j in 1..n where i != j)(x[i] = x[j]) = 0;


% 3.2 --- All Different with index_set(x)

% TODO: Enforce all_different using index_set(x)
constraint sum(i, j in index_set(x) where i<j)(x[i] = x[j])=0;

% -- Part 4 --

% 4.1 --- Output Even Multiples of Three

% Define your output statement in this declaration
array[int] of string: output_step1 ::output_only = [ "Even multiple number \(i) is \(even_multiples_of_three[i])\n" | i in index_set(even_multiples_of_three)]; 

% Uncomment the next link to try your output statement
output output_step1;

% % Output string to give to the checker
string: step1_checker ::output = concat(output_step1);

% 4.2 --- Output Manhattan Distance

% Define your output statement in this declaration
array[int] of string: output_step2 ::output_only =  [ "Distance between \(i) and \(j) is \(manhattan_distance[i,j])\n" 
                                                    | i, j in Point where i < j ];

% Uncomment the next link to try your output statement
output output_step2;

% Output string to give to the checker
string: step2_checker ::output = concat(output_step2);

% 4.3 --- Output Manhattan Distance Table

% Define your output statement in this declaration
array[int] of string: output_step3 ::output_only =  ["\t"] ++ ["\(i)\t" | i in Point] ++ ["\n"] ++ [if j = 1 then "\(i)\t" else "" endif ++ if i < j then "\(manhattan_distance[i,j])\t" else "\t" endif ++ if j = 5 then "\n" endif | i, j in Point];

% Uncomment the next link to try your output statement
output output_step3;

% Output string to give to the checker
string: step3_checker ::output = concat(output_step3);
