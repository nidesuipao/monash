set of int: POS = 1..4;
enum REINDEER = { Lancer, Quentin, Ezekiel, Rudy };
array[POS] of var REINDEER: y;
output ["y = \(y);\n"];

constraint forall(i,j in REINDEER where i < j)(y[i] != y[j]);
constraint forall(i,j in REINDEER where y[i] = Lancer /\ y[j] = Ezekiel)(abs(i-j) != 1);
constraint forall(i,j,k in REINDEER where y[i] = Rudy /\ y[j] = Lancer /\ y[k] = Quentin)(i < j \/ i < k);
