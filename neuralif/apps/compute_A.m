clc;
clear;

format long

n = [144, 576, 2304, 9216, 36864, 147456]; 
cfl = 1.387;

if cfl==0.0277
    Dt = [0.04,0.02,0.01,0.005,0.0025];
end
if cfl==2.77
    Dt = [4,2,1,0.5,0.25,0.125];
end
if cfl==5.55
    Dt = [8,4,2,1,0.5,0.25];
end
if cfl==1.387
    Dt = [2,1,0.5,0.25,0.125,0.0625];
end

for i = 1:length(n)

    M1 = spconvert(load(sprintf('M_ex9_%d_1.387.txt', n(i))));
    K1 = spconvert(load(sprintf('K_ex9_%d_1.387.txt', n(i))));
    dt = Dt(i);

    size(M1)
    size(K1)
    
    A1 = M1 - (dt/2)*K1;
    
    symmetric = issymmetric(A1);
    
    if symmetric
        fprintf('A_ex9_%d_%d is symmetric\n', n(i), cfl);
    else
        fprintf('A_ex9_%d_%d is not symmetric\n', n(i), cfl);
    end

    size(A1);
    [row col v] = find(A1);

    row = round(row, 8,'significant');
    col = round(col, 8,'significant');

    % Write matrix to file
    dlmwrite(sprintf('A_ex9_%d_1.387.txt', n(i)), [row, col, v], ...
        'delimiter', '\t', 'precision', '%.7e');
    
    
    D = diag(A1);
    %full(D)
    size(D);
    
    matrix = K1;
    [row col v] = find(matrix);
    
    for k = 1:length(v)
        if v(k) == 0
            v
        end
    end 

    size(A1)

    % Save spy(A1) plot as an image
    figure;
    spy(A1);
    xlabel('Column');
    ylabel('Row');
    title(sprintf('Spy plot for A (cfl=1.387, n=%d)', n(i)));
    % Get the total number of values shown in the figure
    [numRows, numCols] = size(A1);
    totalValues = nnz(A1);
    fprintf('nnz: %d.\n', totalValues);
    saveas(gcf, sprintf('spy_A_ex9_%d_1.387.png', n(i)));
    close;

end


