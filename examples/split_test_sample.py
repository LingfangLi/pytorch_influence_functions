def split_file(input_file_path, output_dir, total_files=14):
    # 计算每个文件应该包含的行数
    total_lines = 1000
    lines_per_file = total_lines // total_files
    extra_lines = total_lines % total_files

    with open(input_file_path, 'r') as file:
        for file_index in range(1, total_files + 1):
            # 计算当前文件应包含的行数
            current_lines_count = lines_per_file + (1 if file_index <= extra_lines else 0)

            # 生成当前分割文件的路径
            output_file_path = f"{output_dir}/split_{file_index}.txt"

            with open(output_file_path, 'w') as output_file:
                for _ in range(current_lines_count):
                    line = file.readline()
                    # 写入当前行到输出文件
                    output_file.write(line)

                print(f"File {output_file_path} created with {current_lines_count} lines.")


# 调用函数
# input_file_path = r'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\data\src\test1.txt'
# output_dir = 'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\data\src'
# split_file(input_file_path, output_dir)
input_file_path = r'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\data\trg\test1.txt'
output_dir = r'D:\OneDrive - The University of Liverpool\LLMs\InfluenceFunctions\data\trg'
split_file(input_file_path, output_dir)