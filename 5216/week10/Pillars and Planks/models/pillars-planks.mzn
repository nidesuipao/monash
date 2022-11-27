int: planks;
set of int: PLANK = 1..planks;
array[PLANK] of int: plank_width;
int: pillars;
set of int: PILLAR = 1..pillars;
array[PILLAR] of int: pillar_height;
array[PILLAR] of int: pillar_width;

int: available_width;
int: available_height;
set of int: WIDTH = 0..available_width-1;
set of int: HEIGHT = 0..available_height-1; 

array[PLANK] of var WIDTH: xk;
array[PLANK] of var HEIGHT: yk;
array[PILLAR] of var WIDTH: xr;
array[PILLAR] of var HEIGHT: yr;
var 0..available_height: height;

solve minimize height;
%%%% output for grader   
output ["height = \(height);\n"];          
output ["xk = \(xk);\nyk = \(yk);\nxr = \(xr);\nyr = \(yr);\n"]; 

            
constraint height = max(max([yk[p] + 1 | p in PLANK]), max([yr[p] + pillar_height[p] | p in PILLAR])); 

include "diffn.mzn";
constraint diffn(xr ++ xk, yr ++ yk, pillar_width ++ plank_width, pillar_height ++ [1 | p in PLANK]);


constraint forall(p in PLANK)(xk[p] + plank_width[p] <= available_width);

constraint forall(p in PILLAR)(yr[p] + pillar_height[p] <= available_height);
constraint forall(p in PILLAR)(xr[p] + pillar_width[p] <= available_width);

% constraint forall(p in PLANK)(yk[p] != 0);

constraint forall(pk in PLANK)(sum(pr in PILLAR where yr[pr] + pillar_height[pr] = yk[pk])
(xr[pr] <= xk[pk] /\ xr[pr] + pillar_width[pr] > xk[pk] ) = 1);

constraint forall(pk in PLANK)(sum(pr in PILLAR where yr[pr] + pillar_height[pr] = yk[pk])
( xr[pr] < xk[pk] + plank_width[pk] /\ xr[pr] + pillar_width[pr] >= xk[pk] + plank_width[pk] ) = 1 );

% % constraint forall(pk in PLANK)(sum(pr in PILLAR where xk[pk] + plank_width[pk] = xr[pr] + pillar_width[pr])(yr[pr] + pillar_height[pr] = yk[pk]) = 1);

constraint forall(pr in PILLAR)(yr[pr] = 0 \/ sum(pk in PLANK where xr[pr] >= xk[pk] /\ xr[pr] + pillar_width[pr] <= xk[pk] + plank_width[pk])
(yk[pk] + 1 = yr[pr]) = 1);



% each plank is supported at both ends by a pillar

% array[PLANK] of var PILLAR: left;
% array[PLANK] of var PILLAR: right;
% constraint forall(p in PLANK)( xr[left[p]] <= xk[p] /\ xr[left[p]] + pillar_width[left[p]] > xk[p] 
%                                /\ yk[p] = yr[left[p]] + pillar_height[left[p]] );
                               
% constraint forall(p in PLANK)( xr[right[p]] <= xk[p] + plank_width[p] - 1 /\ xr[right[p]] + pillar_width[right[p]] > xk[p] + plank_width[p]-1 
%                                /\ yk[p] = yr[right[p]] + pillar_height[right[p]] );

% each pillar sits on exaclty one plank
% set of int: PLANK0 = 0..planks;
% array[PILLAR] of var 0..planks: support; 
% array[PLANK0] of int: plank_widthx = array1d(PLANK0, [available_width] ++ plank_width);
% array[PLANK0] of var WIDTH: xkx = array1d(PLANK0, [0] ++ xk);
% array[PLANK0] of var -1..available_height-1: ykx = array1d(PLANK0, [-1] ++ yk); 
% constraint forall(p in PILLAR)
%                  (xr[p] >= xkx[support[p]] /\ xr[p] + pillar_width[p] <= xkx[support[p]] + plank_widthx[support[p]]
%                   /\ yr[p] = ykx[support[p]] + 1);











