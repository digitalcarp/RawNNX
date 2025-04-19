// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "imagearea.h"

#include <gtkmm/snapshot.h>

ImageArea::ImageArea() = default;

void ImageArea::measure_vfunc(Gtk::Orientation orientation, int /*for_size*/,
                              int& minimum, int& natural, int& minimum_baseline,
                              int& natural_baseline) const
{
    // NOLINTBEGIN
    if (orientation == Gtk::Orientation::HORIZONTAL) {
        minimum = 400;
        natural = 600;
    } else {
        minimum = 300;
        natural = 400;
    }
    // NOLINTEND

    // Don't use baseline alignment
    minimum_baseline = -1;
    natural_baseline = -1;
}

void ImageArea::snapshot_vfunc(const Glib::RefPtr<Gtk::Snapshot>& snapshot)
{
    const auto allocation = get_allocation();
    const Gdk::Rectangle rect(0, 0, allocation.get_width(), allocation.get_height());

    snapshot->append_color(Gdk::RGBA(1.F, 1.F, 1.F), rect);
}
