#pragma once

/// \file progress.hpp
/// Terminal progress reporting, per Section 9 of the specification.
///
/// Rules this enforces so they cannot be got wrong at a call site: only the
/// root rank ever writes, output is refreshed at most twice a second so a fast
/// iteration cannot spend its time formatting text, and when the destination is
/// not a terminal the bar degrades to occasional plain lines that read sensibly
/// in a CI log.

#include <pnl/core/types.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace pnl {

/// True when standard error is attached to a terminal.
[[nodiscard]] inline bool stderr_is_tty() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    return ::isatty(2) == 1;
#else
    return false;
#endif
}

/// Format a duration in seconds as a compact clock string.
[[nodiscard]] inline std::string format_duration(double seconds) {
    if (!(seconds >= 0.0) || seconds > 359999.0) return "--:--";
    const auto total = static_cast<long>(seconds);
    const long hours = total / 3600;
    const long minutes = (total % 3600) / 60;
    const long secs = total % 60;
    char buffer[32];
    if (hours > 0) {
        std::snprintf(buffer, sizeof(buffer), "%ld:%02ld:%02ld", hours, minutes, secs);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%02ld:%02ld", minutes, secs);
    }
    return buffer;
}

/// A progress bar over a known number of steps.
class ProgressBar {
   public:
    /// \param label shown to the left of the bar.
    /// \param total expected step count; zero means unknown.
    /// \param enabled false on non root ranks, which then print nothing at all.
    ProgressBar(std::string label, Index total, bool enabled)
        : label_(std::move(label)),
          total_(total),
          enabled_(enabled),
          tty_(stderr_is_tty()),
          start_(Clock::now()),
          last_draw_(Clock::now() - std::chrono::seconds(1)) {}

    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&) = delete;
    ProgressBar& operator=(ProgressBar&&) = delete;

    ~ProgressBar() { finish(); }

    /// Report progress. \p detail is appended after the counters, and is where
    /// the iterative solvers put their current residual.
    void update(Index current, std::string_view detail = {}) {
        if (!enabled_) return;
        current_ = current;
        const auto now = Clock::now();
        const auto since = std::chrono::duration<double>(now - last_draw_).count();
        // Twice a second at most, but never skip the final step.
        if (since < MIN_REDRAW_SECONDS && !(total_ > 0 && current >= total_)) return;
        last_draw_ = now;
        draw(detail, false);
    }

    /// Draw the completed bar once and stop. Idempotent.
    void finish(std::string_view detail = {}) {
        if (!enabled_ || finished_) return;
        finished_ = true;
        draw(detail, true);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }

   private:
    using Clock = std::chrono::steady_clock;

    static constexpr double MIN_REDRAW_SECONDS = 0.5;
    static constexpr int BAR_WIDTH = 28;

    void draw(std::string_view detail, bool final_draw) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
        std::string line;
        line.reserve(160);

        if (total_ > 0) {
            const double fraction =
                std::min(1.0, static_cast<double>(current_) / static_cast<double>(total_));
            const int filled = static_cast<int>(fraction * BAR_WIDTH);
            line += label_;
            line += " [";
            for (int i = 0; i < BAR_WIDTH; ++i) line += (i < filled ? '#' : '.');
            line += "] ";
            char counters[96];
            const double eta =
                fraction > 0.0 ? elapsed * (1.0 - fraction) / fraction : 0.0;
            std::snprintf(counters, sizeof(counters), "%3.0f%% %td/%td [%s<%s]",
                          fraction * 100.0, current_, total_,
                          format_duration(elapsed).c_str(), format_duration(eta).c_str());
            line += counters;
        } else {
            char counters[96];
            std::snprintf(counters, sizeof(counters), "%s %td [%s]", label_.c_str(), current_,
                          format_duration(elapsed).c_str());
            line += counters;
        }

        if (!detail.empty()) {
            line += "  ";
            line += detail;
        }

        if (tty_) {
            // Carriage return and clear to end of line, so the bar stays on one
            // line and never leaves fragments of a longer previous line behind.
            std::fprintf(stderr, "\r\033[2K%s", line.c_str());
            std::fflush(stderr);
        } else if (final_draw || current_ == 0 || (total_ > 0 && should_log_step())) {
            // Plain, occasional lines for a CI log.
            std::fprintf(stderr, "%s\n", line.c_str());
            std::fflush(stderr);
        }
    }

    /// In non terminal mode, log roughly every ten percent.
    [[nodiscard]] bool should_log_step() const noexcept {
        if (total_ <= 0) return false;
        const Index step = total_ / 10 > 0 ? total_ / 10 : 1;
        return current_ % step == 0;
    }

    std::string label_;
    Index total_;
    Index current_ = 0;
    bool enabled_;
    bool tty_;
    bool finished_ = false;
    Clock::time_point start_;
    Clock::time_point last_draw_;
};

}  // namespace pnl
