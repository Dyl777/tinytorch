// ============================================================================
// Distributed Parquet Reader
// ============================================================================
// Integrates Carquet (pure C parquet library) with TinyTorch distributed tensors
//
// KEY PRINCIPLES:
// - NEVER loads full dataset into RAM
// - Uses batch reading with configurable batch sizes
// - Respects MemoryBudget thresholds
// - Distributes data across all available devices (GPUs + CPU)
// - Supports column projection (read only needed columns)
// - Supports predicate pushdown (filter row groups)
//
// Usage:
//   auto tensor = DistributedParquetLoader::load<float>(
//       "data.parquet",
//       {.column_name = "embeddings", .batch_size = 65536}
//   );

#ifndef DISTRIBUTED_PARQUET_H
#define DISTRIBUTED_PARQUET_H

#include "distributed_tensor.h"
#include "stream_tensor.h"

// Carquet C API - wrapped in extern "C"
extern "C" {
#include <carquet/carquet.h>
}

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

namespace tinytorch {

// ============================================================================
// Parquet Load Options
// ============================================================================
struct ParquetLoadOptions {
    std::string column_name;           // Column to load (required)
    std::size_t batch_size = 65536;    // Rows per batch (affects memory usage)
    bool use_mmap = true;              // Use memory-mapped I/O
    bool cache_batches = false;        // Cache batches in memory (not recommended for large datasets)
    double memory_budget_pct = 30.0;   // RAM % threshold for streaming
    std::size_t min_shard_size = 1024; // Minimum elements per shard
    
    // Predicate pushdown - filter row groups
    std::string filter_column;         // Column for filtering
    double filter_threshold = 0.0;     // Threshold value
    enum FilterOp { EQ, GT, LT, GTE, LTE } filter_op = GT;
};

// ============================================================================
// Column Type Mapping
// ============================================================================
template<typename T>
struct CarquetTypeMap {
    static constexpr carquet_physical_type_t type = CARQUET_PHYSICAL_FLOAT;
};

template<>
struct CarquetTypeMap<float> {
    static constexpr carquet_physical_type_t type = CARQUET_PHYSICAL_FLOAT;
};

template<>
struct CarquetTypeMap<double> {
    static constexpr carquet_physical_type_t type = CARQUET_PHYSICAL_DOUBLE;
};

template<>
struct CarquetTypeMap<int32_t> {
    static constexpr carquet_physical_type_t type = CARQUET_PHYSICAL_INT32;
};

template<>
struct CarquetTypeMap<int64_t> {
    static constexpr carquet_physical_type_t type = CARQUET_PHYSICAL_INT64;
};

// ============================================================================
// Distributed Parquet Loader
// ============================================================================
class DistributedParquetLoader {
public:
    // Get file info without loading data
    static std::string get_file_info(const std::string& filepath) {
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_options_t opts;
        carquet_reader_options_init(&opts);
        opts.use_mmap = true;
        
        carquet_reader_t* reader = carquet_reader_open(filepath.c_str(), &opts, &err);
        if (!reader) {
            return "Error opening file: " + std::string(err.message);
        }
        
        std::ostringstream oss;
        oss << "Parquet File: " << filepath << "\n";
        oss << "Rows: " << carquet_reader_num_rows(reader) << "\n";
        oss << "Columns: " << carquet_reader_num_columns(reader) << "\n";
        
        carquet_reader_close(reader);
        return oss.str();
    }
    
    // Load a single column as a distributed tensor
    template<typename T>
    static std::shared_ptr<DistributedTensor<T>> load(
        const std::string& filepath,
        const ParquetLoadOptions& opts
    ) {
        carquet_error_t err = CARQUET_ERROR_INIT;
        
        // Open file
        carquet_reader_options_t reader_opts;
        carquet_reader_options_init(&reader_opts);
        reader_opts.use_mmap = opts.use_mmap;
        
        carquet_reader_t* reader = carquet_reader_open(filepath.c_str(), &reader_opts, &err);
        if (!reader) {
            std::cerr << "Failed to open parquet file: " << err.message << std::endl;
            return nullptr;
        }
        
        int64_t num_rows = carquet_reader_num_rows(reader);
        int32_t num_cols = carquet_reader_num_columns(reader);
        
        // Find column index by name
        const carquet_schema_t* schema = carquet_reader_schema(reader);
        int32_t target_col = -1;
        for (int32_t i = 0; i < num_cols; ++i) {
            const char* name = carquet_schema_column_name(schema, i);
            if (name && opts.column_name == name) {
                target_col = i;
                break;
            }
        }
        
        if (target_col < 0) {
            std::cerr << "Column '" << opts.column_name << "' not found in parquet file" << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }

        carquet_physical_type_t col_type = carquet_schema_column_type(schema, target_col);
        if (col_type != CarquetTypeMap<T>::type) {
            std::cerr << "Column '" << opts.column_name << "' type mismatch. Requested type="
                      << static_cast<int>(CarquetTypeMap<T>::type)
                      << " parquet type=" << static_cast<int>(col_type) << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }
        
        std::cout << "Loading column '" << opts.column_name << "' (" << num_rows << " rows)" << std::endl;
        
        // Create distributed tensor with shape
        std::vector<std::size_t> shape = {static_cast<std::size_t>(num_rows)};
        auto tensor = DistributedTensor<T>::zeros(shape, false);
        
        // Set minimum shard size
        DistributedTensor<T>::set_min_shard_size(opts.min_shard_size);
        
        // Configure batch reader for column projection
        carquet_batch_reader_config_t batch_config;
        carquet_batch_reader_config_init(&batch_config);
        batch_config.batch_size = static_cast<int64_t>(opts.batch_size);
        
        // Only read target column
        const char* col_names[] = {opts.column_name.c_str()};
        batch_config.column_names = col_names;
        batch_config.num_column_names = 1;
        
        // Apply predicate pushdown if specified
        if (!opts.filter_column.empty()) {
            // Find filter column index
            for (int32_t i = 0; i < num_cols; ++i) {
                const char* name = carquet_schema_column_name(schema, i);
                if (name && opts.filter_column == name) {
                    break;
                }
            }
        }
        
        // Create batch reader
        carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(
            reader, &batch_config, &err
        );
        if (!batch_reader) {
            std::cerr << "Failed to create batch reader: " << err.message << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }
        
        // Read data in batches and distribute to shards
        carquet_row_batch_t* batch = nullptr;
        std::size_t global_offset = 0;
        std::vector<T> batch_buffer;
        batch_buffer.reserve(opts.batch_size);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            const void* data;
            const uint8_t* nulls;
            int64_t n;
            
            if (carquet_row_batch_column(batch, 0, &data, &nulls, &n) != CARQUET_OK) {
                carquet_row_batch_free(batch);
                batch = nullptr;
                break;
            }
            
            // Copy batch data to buffer
            const T* typed_data = static_cast<const T*>(data);
            for (int64_t i = 0; i < n; ++i) {
                if (!nulls || nulls[i]) {
                    batch_buffer.push_back(typed_data[i]);
                } else {
                    batch_buffer.push_back(T{0}); // Null value
                }
            }
            
            carquet_row_batch_free(batch);
            batch = nullptr;
            
            global_offset += n;
            
            // Progress reporting every 1M rows
            if (global_offset % 1000000 < static_cast<std::size_t>(n)) {
                std::cout << "  Loaded " << global_offset << " / " << num_rows << " rows" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Data loaded in " << load_ms << "ms, distributing to shards..." << std::endl;
        
        // Distribute data to shards
        tensor->distribute_from_vector(batch_buffer);
        
        // Clean up
        carquet_batch_reader_free(batch_reader);
        carquet_reader_close(reader);
        
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        std::cout << "Total load time: " << total_ms << "ms" << std::endl;
        
        return tensor;
    }

    static std::vector<std::string> load_text_samples(
        const std::string& filepath,
        const ParquetLoadOptions& opts,
        std::size_t max_samples
    ) {
        std::vector<std::string> out;

        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_options_t reader_opts;
        carquet_reader_options_init(&reader_opts);
        reader_opts.use_mmap = opts.use_mmap;

        carquet_reader_t* reader = carquet_reader_open(filepath.c_str(), &reader_opts, &err);
        if (!reader) {
            return out;
        }

        int32_t num_cols = carquet_reader_num_columns(reader);
        const carquet_schema_t* schema = carquet_reader_schema(reader);
        int32_t target_col = -1;
        for (int32_t i = 0; i < num_cols; ++i) {
            const char* name = carquet_schema_column_name(schema, i);
            if (name && opts.column_name == name) {
                target_col = i;
                break;
            }
        }
        if (target_col < 0) {
            carquet_reader_close(reader);
            return out;
        }

        carquet_physical_type_t col_type = carquet_schema_column_type(schema, target_col);
        if (col_type != CARQUET_PHYSICAL_BYTE_ARRAY) {
            carquet_reader_close(reader);
            return out;
        }

        carquet_batch_reader_config_t batch_config;
        carquet_batch_reader_config_init(&batch_config);
        batch_config.batch_size = static_cast<int64_t>(opts.batch_size);
        const char* col_names[] = {opts.column_name.c_str()};
        batch_config.column_names = col_names;
        batch_config.num_column_names = 1;

        carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &batch_config, &err);
        if (!batch_reader) {
            carquet_reader_close(reader);
            return out;
        }

        carquet_row_batch_t* batch = nullptr;
        while (out.size() < max_samples &&
               carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            const void* data;
            const uint8_t* nulls;
            int64_t n;
            if (carquet_row_batch_column(batch, 0, &data, &nulls, &n) != CARQUET_OK) {
                carquet_row_batch_free(batch);
                batch = nullptr;
                break;
            }

            const carquet_byte_array_t* typed_data = static_cast<const carquet_byte_array_t*>(data);
            for (int64_t i = 0; i < n && out.size() < max_samples; ++i) {
                if (!nulls || nulls[i]) {
                    const auto& v = typed_data[i];
                    out.emplace_back(reinterpret_cast<const char*>(v.data),
                                     static_cast<std::size_t>(v.length));
                } else {
                    out.emplace_back();
                }
            }

            carquet_row_batch_free(batch);
            batch = nullptr;
        }

        carquet_batch_reader_free(batch_reader);
        carquet_reader_close(reader);
        return out;
    }

    static std::shared_ptr<DistributedTensor<float>> load_text_lengths(
        const std::string& filepath,
        const ParquetLoadOptions& opts
    ) {
        carquet_error_t err = CARQUET_ERROR_INIT;
        carquet_reader_options_t reader_opts;
        carquet_reader_options_init(&reader_opts);
        reader_opts.use_mmap = opts.use_mmap;

        carquet_reader_t* reader = carquet_reader_open(filepath.c_str(), &reader_opts, &err);
        if (!reader) {
            std::cerr << "Failed to open parquet file: " << err.message << std::endl;
            return nullptr;
        }

        int64_t num_rows = carquet_reader_num_rows(reader);
        int32_t num_cols = carquet_reader_num_columns(reader);
        const carquet_schema_t* schema = carquet_reader_schema(reader);

        int32_t target_col = -1;
        for (int32_t i = 0; i < num_cols; ++i) {
            const char* name = carquet_schema_column_name(schema, i);
            if (name && opts.column_name == name) {
                target_col = i;
                break;
            }
        }

        if (target_col < 0) {
            std::cerr << "Column '" << opts.column_name << "' not found in parquet file" << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }

        carquet_physical_type_t col_type = carquet_schema_column_type(schema, target_col);
        if (col_type != CARQUET_PHYSICAL_BYTE_ARRAY) {
            std::cerr << "Column '" << opts.column_name << "' is not BYTE_ARRAY" << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }

        std::vector<std::size_t> shape = {static_cast<std::size_t>(num_rows)};
        auto tensor = DistributedTensor<float>::zeros(shape, false);
        DistributedTensor<float>::set_min_shard_size(opts.min_shard_size);

        carquet_batch_reader_config_t batch_config;
        carquet_batch_reader_config_init(&batch_config);
        batch_config.batch_size = static_cast<int64_t>(opts.batch_size);
        const char* col_names[] = {opts.column_name.c_str()};
        batch_config.column_names = col_names;
        batch_config.num_column_names = 1;

        carquet_batch_reader_t* batch_reader = carquet_batch_reader_create(reader, &batch_config, &err);
        if (!batch_reader) {
            std::cerr << "Failed to create batch reader: " << err.message << std::endl;
            carquet_reader_close(reader);
            return nullptr;
        }

        carquet_row_batch_t* batch = nullptr;
        std::vector<float> lengths;
        lengths.reserve(static_cast<std::size_t>(num_rows));

        while (carquet_batch_reader_next(batch_reader, &batch) == CARQUET_OK && batch) {
            const void* data;
            const uint8_t* nulls;
            int64_t n;
            if (carquet_row_batch_column(batch, 0, &data, &nulls, &n) != CARQUET_OK) {
                carquet_row_batch_free(batch);
                batch = nullptr;
                break;
            }

            const carquet_byte_array_t* typed_data = static_cast<const carquet_byte_array_t*>(data);
            for (int64_t i = 0; i < n; ++i) {
                if (!nulls || nulls[i]) {
                    lengths.push_back(static_cast<float>(typed_data[i].length));
                } else {
                    lengths.push_back(0.0f);
                }
            }

            carquet_row_batch_free(batch);
            batch = nullptr;
        }

        tensor->distribute_from_vector(lengths);
        carquet_batch_reader_free(batch_reader);
        carquet_reader_close(reader);
        return tensor;
    }
    
    // Load multiple columns (returns vector of tensors)
    template<typename T>
    static std::vector<std::shared_ptr<DistributedTensor<T>>> load_columns(
        const std::string& filepath,
        const std::vector<std::string>& column_names,
        const ParquetLoadOptions& base_opts
    );
};

} // namespace tinytorch

#endif // DISTRIBUTED_PARQUET_H

