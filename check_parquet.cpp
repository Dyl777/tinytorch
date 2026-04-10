#include <iostream>
#include "parquet_parser/carquet/include/carquet/carquet.h"

int main(int argc, char** argv) {
    const char* filepath = argc > 1 ? argv[1] : "../dataset/train-openweb.parquet";
    
    carquet_error_t err = CARQUET_ERROR_INIT;
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;
    
    carquet_reader_t* reader = carquet_reader_open(filepath, &opts, &err);
    if (!reader) {
        std::cerr << "Failed to open: " << err.message << std::endl;
        return 1;
    }
    
    int64_t num_rows = carquet_reader_num_rows(reader);
    int32_t num_cols = carquet_reader_num_columns(reader);
    
    std::cout << "File: " << filepath << std::endl;
    std::cout << "Rows: " << num_rows << std::endl;
    std::cout << "Columns: " << num_cols << std::endl;
    std::cout << "\nColumn Schema:" << std::endl;
    
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    for (int32_t i = 0; i < num_cols; ++i) {
        const char* name = carquet_schema_column_name(schema, i);
        std::cout << "  [" << i << "] " << (name ? name : "unnamed") << std::endl;
    }
    
    carquet_reader_close(reader);
    return 0;
}
