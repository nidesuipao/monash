% Use this editor as a MiniZinc scratch book
enum PERSON;
array[int,1..2] of PERSON: couple;
array[int,1..2] of PERSON: rivals;

int: n = card(PERSON);
set of int: SEAT = 1..n;
array[SEAT] of var PERSON: person;

var int: _objective;


include "alldifferent.mzn";
constraint alldifferent(person);

constraint forall(p in index_set_1of2(couple))
                 (forall(i,j in SEAT where person[i] = couple[p,1]  /\ person[j] = couple[p,2])
                 (compatible(i,j)));
                 
predicate compatible(var SEAT: i, var SEAT: j) = min((i - j + n) mod n, (j - i + n) mod n) = 1 \/
                                                min((i - j + n) mod n, (j - i + n) mod n) mod n > n div 3 ;
                                                
constraint _objective = sum(p in index_set_1of2(rivals))
                 (forall(i,j in SEAT where person[i] = rivals[p,1]  /\ person[j] = rivals[p,2])
                 (compatible(i,j))); 
                 

output["\(person)\n \(_objective)"]