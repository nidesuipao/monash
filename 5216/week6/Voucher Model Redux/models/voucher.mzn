% free pizza problem
%
% pizzas to order
%    13 25 17 12 9
% vouchers
%    1+1  2+1
% you pay for the most expensive pizzas when using a voucher.

int: n; % number of pizzas
set of int: PIZZA = 1..n;
array[PIZZA] of int: price;   % price of each pizza

int: m; % number of vouchers
set of int: VOUCHER = 1..m;
array[VOUCHER] of int: buy;   % buy this many to use voucher
array[VOUCHER] of int: free;  % get this many free

set of int: ASSIGN = -m .. m; % -i pizza is assigned to buy of voucher i
                              %  i pizza is assigned to free of voucher i
                              %  0 no voucher used on pizza

array[PIZZA] of var ASSIGN: how;
array[VOUCHER] of var bool: used;

% assign right number of pizzas to buy order
% constraint forall(v in VOUCHER)(used[v] -> sum(p in PIZZA)(how[p] = -v) = buy[v]);
constraint forall(p in PIZZA)(used[abs(how[p])] = true);
constraint forall(v in VOUCHER)(sum(p in PIZZA)(how[p] = -v) = used[v]*buy[v]);

% assign not too many pizzas to free order
constraint forall(v in VOUCHER)(sum(p in PIZZA)(how[p] = v) = used[v]*free[v]);

% pizzas assigned to free are no more expensive than pizzas assigned to buy
array[VOUCHER] of var int: maxp;
array[VOUCHER] of var int: minp;
constraint forall(v in VOUCHER)(forall(p in PIZZA where how[p] = v)(maxp[v] = max(price[p], maxp[v])));
constraint forall(v in VOUCHER)(forall(p in PIZZA where how[p] = -v)(minp[v] = min(price[p], minp[v])));
constraint forall(v in VOUCHER)(maxp[v] <= minp[v]);
% constraint forall(p1, p2 in PIZZA)(how[p1] = -how[p2] /\ how[p1] < 0 -> price[p2] <= price[p1]);



int: total = sum(price);
var 0..total: cost = sum(p in PIZZA)((how[p] <= 0)*price[p]);

solve minimize cost;

output  ["price = \(price);\nbuy = \(buy);\nfree = \(free);\nhow = \(how);\ncost = \(cost);\nused = \(maxp);\n"];
% output  ["price = \(price);\nbuy = \(buy);\nfree = \(free);\nhow = \(how);\ncost = \(cost);\n"];
