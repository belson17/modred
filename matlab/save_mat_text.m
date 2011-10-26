function save_mat_text(mat, filename, delimiter)
%Saves a matrix to filename.
%   Matches the same format as in modaldecomp's save_mat_text.
    
    [nrows,ncols] = size(mat);
    is_complex = ~isreal(mat);
    
    if is_complex ~= 0
        mat_real = real(mat);
        mat_imag = imag(mat);
        ncols = 2*ncols;
        mat_write = zeros(nrows,ncols);
        mat_write(:,1:2:end-1) = mat_real(:,:);
        mat_write(:,2:2:end) = mat_imag(:,:);
    else
        mat_write = mat;
    end
    
    file_ID = fopen(filename);
    format = '%.16f';
    for row=1:nrows
        fprintf(file_ID, [format, delimiter],mat_write(row,1:end-1));
        fprintf(file_ID, [format, '\n'],mat_write(row,end));
    end
end

