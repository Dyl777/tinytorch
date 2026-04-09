#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * GnuplotPipe: Synchronous gnuplot wrapper for real-time plotting.
 * 
 * Uses _popen/popen to pipe commands to gnuplot, enabling live updates.
 * Call plotLine() repeatedly with growing datasets for synchronous live plotting.
 */
class GnuplotPipe {
private:
    FILE* pipe;
    std::string plot_title;
    std::string x_label;
    std::string y_label;
    bool is_initialized;
    std::vector<std::string> setup_commands;

    /**
     * Send a command to gnuplot.
     */
    void send(const std::string& cmd) {
        if (pipe == nullptr || !is_initialized) {
            throw std::runtime_error("Gnuplot pipe is not open.");
        }
        fprintf(pipe, "%s\n", cmd.c_str());
        fflush(pipe);
    }

    /**
     * Find gnuplot executable (Windows).
     */
#ifdef _WIN32
    static std::string findGnuplot() {
        // Prefer gnuplot.exe (console version that supports piping to GUI terminal)
        const char* paths[] = {
            "gnuplot.exe",
            "C:\\Program Files\\gnuplot\\bin\\gnuplot.exe",
            "C:\\Program Files (x86)\\gnuplot\\bin\\gnuplot.exe",
            "wgnuplot_pipes.exe",
            "C:\\Program Files\\gnuplot\\bin\\wgnuplot_pipes.exe",
        };
        
        for (const char* path : paths) {
            // Check direct path
            DWORD attrs = GetFileAttributesA(path);
            if (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
                return path;
            }
            
            // Search in PATH for simple names
            std::string check = path;
            if (check.find('\\') == std::string::npos) {
                char buffer[MAX_PATH];
                if (SearchPathA(nullptr, path, nullptr, MAX_PATH, buffer, nullptr) > 0) {
                    return path;
                }
            }
        }
        
        throw std::runtime_error(
            "gnuplot not found. Please install gnuplot.\n"
            "Download: https://sourceforge.net/projects/gnuplot/files/gnuplot/"
        );
    }
#endif

public:
    /**
     * Constructor: Opens gnuplot with persistent window.
     */
    GnuplotPipe(const std::string& title = "Real-time Plot",
                const std::string& xlabel = "X",
                const std::string& ylabel = "Y")
        : pipe(nullptr), plot_title(title), x_label(xlabel), y_label(ylabel), is_initialized(false) {
        
        std::string gnuplot_exe;
#ifdef _WIN32
        gnuplot_exe = findGnuplot();
        // Use -persist to keep window open after we close the pipe
        pipe = _popen(("\"" + gnuplot_exe + "\" -persist").c_str(), "w");
#else
        pipe = popen("gnuplot -persist", "w");
#endif
        
        if (pipe == nullptr) {
            throw std::runtime_error("Failed to start gnuplot. Is it installed?");
        }
        
        is_initialized = true;
        
        // Set up terminal and labels
#ifdef _WIN32
        send("set terminal windows");
#endif
        send("set title \"" + plot_title + "\"");
        send("set xlabel \"" + x_label + "\"");
        send("set ylabel \"" + y_label + "\"");
        send("set grid");
    }

    /**
     * Plot data synchronously. Blocks until gnuplot processes the data.
     * Call this repeatedly with growing vectors for live updates.
     */
    void plotLine(const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::string& title = "data") {
        if (!is_initialized || x.empty() || y.empty()) return;
        if (x.size() != y.size()) {
            throw std::invalid_argument("x and y must have same size");
        }

        // Plot with inline data
        send("plot '-' using 1:2 with linespoints title '" + title + "'");
        
        for (size_t i = 0; i < x.size(); ++i) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.10g %.10g", x[i], y[i]);
            send(buf);
        }
        // End-of-data marker
        send("e");
    }

    /**
     * Plot multiple lines simultaneously.
     */
    void plotMultipleLines(
        const std::vector<std::tuple<std::vector<double>, std::vector<double>, std::string>>& datasets) {
        if (!is_initialized || datasets.empty()) return;

        // Build plot command
        std::string cmd = "plot ";
        for (size_t i = 0; i < datasets.size(); ++i) {
            cmd += "'-' using 1:2 with linespoints title '" + std::get<2>(datasets[i]) + "'";
            if (i < datasets.size() - 1) cmd += ", ";
        }
        send(cmd);

        // Send each dataset
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

    /**
     * Set axis ranges.
     */
    void setRanges(double xmin, double xmax, double ymin, double ymax) {
        char buf[128];
        snprintf(buf, sizeof(buf), "set xrange [%g:%g]", xmin, xmax);
        send(buf);
        snprintf(buf, sizeof(buf), "set yrange [%g:%g]", ymin, ymax);
        send(buf);
    }

    /**
     * Set gnuplot option.
     */
    void setOption(const std::string& opt) {
        send(opt);
    }

    /**
     * Clear plot.
     */
    void clear() {
        send("clear");
    }

    /**
     * Destructor: Close pipe.
     */
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
