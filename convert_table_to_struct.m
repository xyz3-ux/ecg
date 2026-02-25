function convert_table_to_struct(input_file, output_file)
    % Load the data
    data = load(input_file);
    
    % Get all variable names
    vars = fieldnames(data);
    
    % Loop through variables to find the table
    found_table = false;
    for i = 1:numel(vars)
        if istable(data.(vars{i}))
            T = data.(vars{i});
            found_table = true;
            
            % --- SANITIZATION STEPS ---
            
            % 1. Identify Categorical Columns and convert to Cell-Strings
            % Python hates MATLAB categoricals; it loves strings.
            var_names = T.Properties.VariableNames;
            for j = 1:numel(var_names)
                col_name = var_names{j};
                if iscategorical(T.(col_name))
                    T.(col_name) = cellstr(T.(col_name));
                end
            end
            
            % 2. Convert Table to Scalar Struct (Column-Oriented)
            % 'ToScalar', true creates a struct where fields are entire columns.
            % This is vastly faster for Python to read than row-by-row structs.
            clean_struct = table2struct(T, 'ToScalar', true);
            
            % Save the clean struct to the new file
            % -v7 is faster for scipy/pymatreader than -v7.3 for files < 2GB
            save(output_file, '-struct', 'clean_struct', '-v7');
            
            fprintf('Processed: %s -> %s\n', input_file, output_file);
            break; 
        end
    end
    
    if ~found_table
        warning('No table found in %s', input_file);
    end
end