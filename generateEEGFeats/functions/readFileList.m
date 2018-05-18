function file_list = readFileList(filename)
    fid = fopen(filename);

    i=1;
    file_line = fgetl(fid);
    while ischar(file_line)
        file_list{i} = file_line;
        i=i+1;
        file_line = fgetl(fid);
    end
    fclose(fid);
end