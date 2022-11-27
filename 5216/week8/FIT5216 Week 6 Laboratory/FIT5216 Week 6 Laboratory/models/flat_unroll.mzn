array[1..4] of int: a = [1,4,6,3];
array[1..4] of int: b = [3,2,1,4];
array[1..4] of var 0..9: x;
constraint forall(i in 1..4 where a[i] < b[i])(x[i] >= 4);
constraint sum(i in 2..4)((a[i] - b[i]) * x[i]) >= 37;
constraint forall(i in 1..4)(x[1] <= x[i]);

solve :: int_search(x, input_order, indomain_min, complete)
%      minimize sum(x);
satisfy;

