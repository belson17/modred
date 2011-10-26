function [ mat ] = load_mat_text(filename, is_complex)
% Loads a text matrix file. 
% is_complex is 0 for real data, 1 for complex. Corresponds to the 
% functions in modaldecomp.

    mat_raw = load(filename);
    % The cols alternate real, imag for complex data
    [nrows, ncols] = size(mat_raw);
    if is_complex ~= 0
        if mod(ncols,2) ~= 0
            error('Did not have even number of columns for complex data')
        end
        mat_real = mat_raw(:,1:2:end-1);
        mat_imag = mat_raw(:,2:2:end);
        mat = mat_real+1i*mat_imag;
    else
        mat = mat_raw;
    end
    
end

