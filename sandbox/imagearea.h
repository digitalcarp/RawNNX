// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#pragma once

#include <gtkmm/widget.h>

class ImageArea : public Gtk::Widget {
public:
    ImageArea();

protected:
    void measure_vfunc(Gtk::Orientation orientation, int for_size, int& minimum,
                       int& natural, int& minimum_baseline,
                       int& natural_baseline) const override;
    void snapshot_vfunc(const Glib::RefPtr<Gtk::Snapshot>& snapshot) override;
};
