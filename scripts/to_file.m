v_filename = sprintf('../data/%s_y_H%.2f.bin', filename, h);
fileID = fopen(v_filename, 'w');
fwrite(fileID, y', 'double');
fclose(fileID);