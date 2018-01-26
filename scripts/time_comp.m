figure
matTime = 2.0857;
shTime = 0.014011;
glTime = 0.033040;

A = [matTime; shTime; glTime];
names = categorical({'Matlab', 'MemShared', 'MemGlobal'});
%names = [1 2 3];
bar(names, A)
title("Time comparison (r15, ó^2=25 (one final point), å = 10^{-9})")
ylabel("Time (s)")

figure
B = [matTime/shTime matTime/glTime];
names = categorical({'MemShared', 'MemGlobal'});
bar(names, B)
title('Speed up: Cuda vs Matlab (r15, ó^2=25, å = 10^{-9})');
ylabel("Speed up");

figure
matTime =  0.2082;
shTime = 0.007347;
glTime = 0.017918;

A = [matTime; shTime; glTime];
names = categorical({'Matlab', 'MemShared', 'MemGlobal'});
%names = [1 2 3];
bar(names, A)
title("Time comparison (r15, ó^2=1 (15 final points), å = 10^{-9})")
ylabel("Time (s)")

figure
B = [matTime/shTime matTime/glTime];
names = categorical({'MemShared', 'MemGlobal'});
bar(names, B)
title('Speed up: Cuda vs Matlab (r15, ó^2=1, å = 10^{-9})');
ylabel("Speed up");

figure
matTime =  0.4711;
shTime = 0.006070;
glTime = 0.010335;

A = [matTime; shTime; glTime];
names = categorical({'Matlab', 'MemShared', 'MemGlobal'});
%names = [1 2 3];
bar(names, A)
title("Time comparison (seeds, ó^2=3, å = 10^{-9})")
ylabel("Time (s)")

figure
B = [matTime/shTime matTime/glTime];
names = categorical({'MemShared', 'MemGlobal'});
bar(names, B)
title('Speed up: Cuda vs Matlab (seeds, ó^2=3, å = 10^{-9})');
ylabel("Speed up");

