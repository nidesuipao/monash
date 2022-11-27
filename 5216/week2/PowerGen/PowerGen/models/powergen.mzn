% enum 
% power generation
int: T;                 % decades
array[1..T] of int: e;  % expected requirements
array[1..T] of int: a;  % current production

array[1..T] of var 0..infinity: N; % number of nuclear power plants built each decade
array[1..T] of var 0..infinity: C; % number of coal power plants built each decade
array[1..T] of var 0..infinity: S; % number of solar power plants built each decade

var 0..infinity: cost;  % costs of building all new power plants
solve minimize cost;

constraint forall(i in 1..T)(
  e[i] - a[i] <= sum(t in max(i-5,1)..i)(N[t]*4) + sum(t in max(i-1,1)..i)(C[t]) + sum(t in max(i-2,1)..i)(S[t])
  );
 
constraint forall(i in 1..T)(
  10 * sum(t in max(i-5,1)..i)(N[t] * 4) <= 4 * (a[i] + sum(t in max(i-5,1)..i)(N[t]*4) + sum(t in max(i-1,1)..i)(C[t]) + sum(t in max(i-2,1)..i)(S[t]))
  );
  
constraint forall(i in 1..T)(
  10 * sum(t in max(i-2,1)..i)(S[t]) >= 2 * (a[i] + sum(t in max(i-5,1)..i)(N[t]*4) + sum(t in max(i-1,1)..i)(C[t]) + sum(t in max(i-2,1)..i)(S[t]))
  );

constraint(sum(t in 1..T)(C[t]) <= 10);

cost = sum(i in 1..T)(N[i]*10 + C[i]*1 + S[i]*2)

                           
