#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================
// ImageWriter: Minimal BMP file writer (no external deps)
// ============================================================
class ImageWriter {
public:
    /**
     * Save a 24-bit BMP file from RGB pixel data.
     * @param filename: Output file path
     * @param width, height: Image dimensions
     * @param pixels: Row-major RGB data (3 bytes per pixel)
     */
    static bool saveBMP(const std::string& filename, int width, int height,
                        const std::vector<uint8_t>& pixels) {
        if (pixels.size() != (size_t)width * height * 3) return false;

#ifdef _WIN32
        FILE* f = fopen(filename.c_str(), "wb");
#else
        FILE* f = fopen(filename.c_str(), "wb");
#endif
        if (!f) return false;

        // BMP rows are bottom-to-top, each row padded to 4-byte boundary
        int row_padded = (width * 3 + 3) & (~3);
        int file_size = 54 + row_padded * height;

        // BMP File Header (14 bytes)
        uint8_t bmp_header[14] = {
            'B', 'M',                         // Signature
            0, 0, 0, 0,                       // File size (filled below)
            0, 0, 0, 0,                       // Reserved
            54, 0, 0, 0                       // Pixel data offset
        };
        bmp_header[2] = file_size & 0xFF;
        bmp_header[3] = (file_size >> 8) & 0xFF;
        bmp_header[4] = (file_size >> 16) & 0xFF;
        bmp_header[5] = (file_size >> 24) & 0xFF;

        // DIB Header (BITMAPINFOHEADER, 40 bytes)
        uint8_t dib_header[40] = {0};
        dib_header[0] = 40;                   // Header size
        // Width (4 bytes, little-endian)
        dib_header[4] = width & 0xFF;
        dib_header[5] = (width >> 8) & 0xFF;
        dib_header[6] = (width >> 16) & 0xFF;
        dib_header[7] = (width >> 24) & 0xFF;
        // Height (4 bytes)
        dib_header[8] = height & 0xFF;
        dib_header[9] = (height >> 8) & 0xFF;
        dib_header[10] = (height >> 16) & 0xFF;
        dib_header[11] = (height >> 24) & 0xFF;
        dib_header[12] = 1;                   // Color planes
        dib_header[14] = 24;                  // Bits per pixel

        fwrite(bmp_header, 1, 14, f);
        fwrite(dib_header, 1, 40, f);

        // Write pixel data (bottom-to-top, BGR order)
        std::vector<uint8_t> padding(row_padded - width * 3, 0);
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                uint8_t bgr[3] = {
                    pixels[idx + 2],  // B
                    pixels[idx + 1],  // G
                    pixels[idx + 0]   // R
                };
                fwrite(bgr, 1, 3, f);
            }
            fwrite(padding.data(), 1, padding.size(), f);
        }

        fclose(f);
        return true;
    }
};

// ============================================================
// GnuplotPipe: Synchronous gnuplot wrapper for real-time plotting
// ============================================================

class GnuplotPipe {
private:
    FILE* pipe;
    std::string plot_title;
    std::string x_label;
    std::string y_label;
    bool is_initialized;
    int window_id;
    bool terminal_windows_set;

    void send(const std::string& cmd) {
        if (pipe == nullptr || !is_initialized) {
            throw std::runtime_error("Gnuplot pipe is not open.");
        }
        fprintf(pipe, "%s\n", cmd.c_str());
        fflush(pipe);
    }

#ifdef _WIN32
    static std::string findGnuplot() {
        const char* paths[] = {
            "gnuplot.exe",
            "C:\\Program Files\\gnuplot\\bin\\gnuplot.exe",
            "C:\\Program Files (x86)\\gnuplot\\bin\\gnuplot.exe",
            "wgnuplot_pipes.exe",
            "C:\\Program Files\\gnuplot\\bin\\wgnuplot_pipes.exe",
        };

        for (const char* path : paths) {
            DWORD attrs = GetFileAttributesA(path);
            if (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
                return path;
            }
            std::string check = path;
            if (check.find('\\') == std::string::npos) {
                char buffer[MAX_PATH];
                if (SearchPathA(nullptr, path, nullptr, MAX_PATH, buffer, nullptr) > 0) {
                    return path;
                }
            }
        }
        throw std::runtime_error(
            "gnuplot not found. Install gnuplot.\n"
            "Download: https://sourceforge.net/projects/gnuplot/files/gnuplot/"
        );
    }
#endif

public:
    /**
     * Opens a gnuplot window for live plotting.
     * @param title: Plot title
     * @param xlabel: X-axis label
     * @param ylabel: Y-axis label
     * @param win_id: Window slot number (0-9) for multiple simultaneous windows
     */
    GnuplotPipe(const std::string& title = "Plot",
                const std::string& xlabel = "X",
                const std::string& ylabel = "Y",
                int win_id = 0)
        : pipe(nullptr), plot_title(title), x_label(xlabel), y_label(ylabel),
          is_initialized(false), window_id(win_id), terminal_windows_set(false) {

        std::string gnuplot_exe;
#ifdef _WIN32
        gnuplot_exe = findGnuplot();
        pipe = _popen(("\"" + gnuplot_exe + "\" -persist").c_str(), "w");
#else
        pipe = popen("gnuplot -persist", "w");
#endif

        if (pipe == nullptr) {
            throw std::runtime_error("Failed to start gnuplot.");
        }

        is_initialized = true;

        // Set terminal to a specific window for multi-window support
#ifdef _WIN32
        std::string term_cmd = "set terminal windows title \"" + title + "\"";
        send(term_cmd);
        terminal_windows_set = true;
#else
        send("set terminal x11 " + std::to_string(win_id) + " title \"" + title + "\"");
#endif

        send("set xlabel \"" + x_label + "\"");
        send("set ylabel \"" + y_label + "\"");
        send("set grid");
    }

    /**
     * Plot a single line synchronously.
     */
    void plotLine(const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::string& title = "data") {
        if (!is_initialized || x.empty() || y.empty()) return;
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have same size");
        }

        send("plot '-' using 1:2 with linespoints title '" + title + "'");
        for (size_t i = 0; i < x.size(); ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.10g %.10g", x[i], y[i]);
            send(buf);
        }
        send("e");
    }

    /**
     * Plot multiple lines on the same window.
     */
    void plotMultipleLines(
        const std::vector<std::tuple<std::vector<double>, std::vector<double>, std::string>>& datasets) {
        if (!is_initialized || datasets.empty()) return;

        std::string cmd = "plot ";
        for (size_t i = 0; i < datasets.size(); ++i) {
            cmd += "'-' using 1:2 with linespoints title '" + std::get<2>(datasets[i]) + "'";
            if (i < datasets.size() - 1) cmd += ", ";
        }
        send(cmd);

        for (const auto& ds : datasets) {
            const auto& x = std::get<0>(ds);
            const auto& y = std::get<1>(ds);
            for (size_t i = 0; i < x.size(); ++i) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.10g %.10g", x[i], y[i]);
                send(buf);
            }
            send("e");
        }
    }

    // ── Configuration ──

    void setRanges(double xmin, double xmax, double ymin, double ymax) {
        char buf[128];
        snprintf(buf, sizeof(buf), "set xrange [%g:%g]", xmin, xmax);
        send(buf);
        snprintf(buf, sizeof(buf), "set yrange [%g:%g]", ymin, ymax);
        send(buf);
    }

    void setOption(const std::string& opt) { send(opt); }
    void clear() { send("clear"); }

    // ── Checkpoint: Save current plot as image ──

    /**
     * Save the current plot to a file (PNG, BMP, SVG, etc).
     * Uses gnuplot's `set terminal` + `replot` mechanism.
     * @param filepath: Output file path (extension determines format)
     */
    void saveCheckpoint(const std::string& filepath) {
        if (!is_initialized) return;

        std::string ext;
        size_t dot = filepath.rfind('.');
        if (dot != std::string::npos) ext = filepath.substr(dot + 1);

        // Convert extension to gnuplot terminal name
        std::string term;
        if (ext == "png") term = "pngcairo size 800,600";
        else if (ext == "jpg" || ext == "jpeg") term = "jpeg size 800,600";
        else if (ext == "bmp") term = "bmp size 800,600";
        else if (ext == "svg") term = "svg size 800,600";
        else if (ext == "pdf") term = "pdfcairo size 8,6";
        else if (ext == "eps") term = "postscript eps color";
        else term = "pngcairo size 800,600";  // default

        send("set terminal " + term);
        send("set output \"" + filepath + "\"");
        send("replot");
        send("set output");  // close file

        // Switch back to interactive terminal
#ifdef _WIN32
        if (terminal_windows_set) {
            send("set terminal windows");
        }
#else
        send("set terminal x11");
#endif
    }

    /**
     * Save data to a CSV file for external processing.
     */
    static void saveDataCSV(const std::string& filepath,
                            const std::vector<std::string>& headers,
                            const std::vector<std::vector<double>>& columns) {
        if (headers.size() != columns.size()) {
            throw std::invalid_argument("headers and columns count mismatch");
        }
        if (columns.empty()) return;

        size_t n_rows = columns[0].size();
        for (const auto& col : columns) {
            if (col.size() != n_rows) {
                throw std::invalid_argument("All columns must have same length");
            }
        }

        FILE* f = fopen(filepath.c_str(), "w");
        if (!f) throw std::runtime_error("Cannot write CSV: " + filepath);

        // Header
        for (size_t i = 0; i < headers.size(); ++i) {
            fprintf(f, "%s%s", headers[i].c_str(), i < headers.size() - 1 ? "," : "\n");
        }

        // Data
        for (size_t r = 0; r < n_rows; ++r) {
            for (size_t c = 0; c < columns.size(); ++c) {
                fprintf(f, "%.10g%s", columns[c][r], c < columns.size() - 1 ? "," : "\n");
            }
        }
        fclose(f);
    }

    // ── GIF Generation ──

    /**
     * Generate a GIF animation from a sequence of plot states.
     * Renders each frame as PNG, then assembles into animated GIF.
     *
     * @param frames: Vector of (x_data, y_data) pairs, one per frame
     * @param title: Line title
     * @param output_gif: Output GIF file path
     * @param delay_ms: Frame delay in milliseconds
     * @param clean_up_frames: Delete intermediate PNG files after creating GIF
     */
    void generateGIF(const std::vector<std::tuple<std::vector<double>, std::vector<double>>>& frames,
                     const std::string& title,
                     const std::string& output_gif,
                     int delay_ms = 100,
                     bool clean_up_frames = true) {
        if (!is_initialized || frames.empty()) return;

        std::string temp_dir = output_gif.substr(0, output_gif.rfind('.'));
        int delay_cs = delay_ms / 10;  // gnuplot delay is in centiseconds

#ifdef _WIN32
        // Create temp dir for frames
        CreateDirectoryA(temp_dir.c_str(), nullptr);
#endif

        // Render each frame as PNG
        for (size_t f = 0; f < frames.size(); ++f) {
            const auto& x = std::get<0>(frames[f]);
            const auto& y = std::get<1>(frames[f]);

            if (x.empty()) continue;

            // Switch to PNG terminal for this frame
            std::string frame_file = temp_dir + "/frame_" + std::to_string(f) + ".png";

            send("set terminal pngcairo size 800,600");
            send("set output \"" + frame_file + "\"");
            send("set title \"" + plot_title + " (frame " + std::to_string(f) + "/" + std::to_string(frames.size() - 1) + ")\"");
            send("set grid");

            // Plot this frame's data
            send("plot '-' using 1:2 with linespoints title '" + title + "'");
            for (size_t i = 0; i < x.size(); ++i) {
                char buf[64];
                snprintf(buf, sizeof(buf), "%.10g %.10g", x[i], y[i]);
                send(buf);
            }
            send("e");
            send("set output");
        }

        // Restore interactive terminal
#ifdef _WIN32
        if (terminal_windows_set) {
            send("set terminal windows");
        }
#endif
        send("set title \"" + plot_title + "\"");

        // Generate animated GIF using gnuplot's gif terminal
        // First, create a combined script
        std::string gif_script = temp_dir + "/make_gif.plt";
        FILE* sf = fopen(gif_script.c_str(), "w");
        if (!sf) {
            throw std::runtime_error("Cannot create GIF script");
        }

        fprintf(sf, "set terminal gif animate delay %d size 800,600 optimize\n", delay_cs);
        fprintf(sf, "set output \"%s\"\n", output_gif.c_str());
        fprintf(sf, "set title \"%s\"\n", plot_title.c_str());
        fprintf(sf, "set xlabel \"%s\"\n", x_label.c_str());
        fprintf(sf, "set ylabel \"%s\"\n", y_label.c_str());
        fprintf(sf, "set grid\n");

        for (size_t f = 0; f < frames.size(); ++f) {
            const auto& x = std::get<0>(frames[f]);
            const auto& y = std::get<1>(frames[f]);
            if (x.empty()) continue;

            std::string frame_file = temp_dir + "/frame_" + std::to_string(f) + ".png";
            // We'll re-emit the data inline for each frame
            fprintf(sf, "set title \"%s (frame %zu/%zu)\"\n", plot_title.c_str(), f, frames.size() - 1);
            fprintf(sf, "plot '-' using 1:2 with linespoints title '%s'\n", title.c_str());
            for (size_t i = 0; i < x.size(); ++i) {
                fprintf(sf, "%.10g %.10g\n", x[i], y[i]);
            }
            fprintf(sf, "e\n");
        }

        fclose(sf);

        // Execute the GIF generation
        std::string cmd;
#ifdef _WIN32
        std::string gnuplot_exe = findGnuplot();
        // Escape backslashes for system() and wrap in quotes
        std::string exe_escaped = gnuplot_exe;
        std::string script_escaped = gif_script;
        // Replace forward slashes with backslashes for Windows paths
        for (auto& c : exe_escaped) if (c == '/') c = '\\';
        for (auto& c : script_escaped) if (c == '/') c = '\\';
        cmd = "\"" + exe_escaped + "\" \"" + script_escaped + "\"";
#else
        cmd = "gnuplot \"" + gif_script + "\"";
#endif
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "  Warning: gnuplot GIF generation returned code " << ret << std::endl;
        }

        // Clean up frame PNG files
        if (clean_up_frames) {
            for (size_t f = 0; f < frames.size(); ++f) {
                std::string frame_file = temp_dir + "/frame_" + std::to_string(f) + ".png";
                std::remove(frame_file.c_str());
            }
#ifdef _WIN32
            RemoveDirectoryA(temp_dir.c_str());
#else
            rmdir(temp_dir.c_str());
#endif
        }
        std::remove(gif_script.c_str());

        std::cout << "  GIF saved: " << output_gif << std::endl;
    }

    // ── Cleanup ──

    ~GnuplotPipe() {
        if (is_initialized && pipe) {
            send("exit");
#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif
        }
    }

    GnuplotPipe(const GnuplotPipe&) = delete;
    GnuplotPipe& operator=(const GnuplotPipe&) = delete;
};
