// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "mainwindow.h"

#include "imagearea.h"

#include "fmt/format.h"

#include <giomm/liststore.h>
#include <gtkmm/box.h>
#include <gtkmm/button.h>
#include <gtkmm/error.h>
#include <gtkmm/filedialog.h>
#include <gtkmm/paned.h>
#include <gtkmm/scrolledwindow.h>

MainWindow::MainWindow() : m_img_area(Gtk::make_managed<ImageArea>())
{
    set_title("RawNNX Sandbox");
    set_default_size(800, 600); // NOLINT
    maximize();

    auto* paned = Gtk::make_managed<Gtk::Paned>();
    paned->set_wide_handle();

    paned->set_start_child(*m_img_area);

    auto* scroll = Gtk::make_managed<Gtk::ScrolledWindow>();

    auto* controls = Gtk::make_managed<Gtk::Box>(Gtk::Orientation::VERTICAL);

    auto* choose_image_button = Gtk::make_managed<Gtk::Button>("Load Image");
    choose_image_button->signal_clicked().connect(
            sigc::mem_fun(*this, &MainWindow::onLoadImageButtonClicked));
    controls->append(*choose_image_button);

    scroll->set_child(*controls);
    paned->set_end_child(*scroll);
    paned->set_shrink_end_child();

    set_child(*paned);
}

void MainWindow::onLoadImageButtonClicked()
{
    auto dialog = Gtk::FileDialog::create();

    // Add filters, so that only certain file types can be selected:
    auto filters = Gio::ListStore<Gtk::FileFilter>::create();

    auto filter_jpg = Gtk::FileFilter::create();
    filter_jpg->set_name("JPEG files");
    filter_jpg->add_mime_type("image/jpeg");
    filters->append(filter_jpg);

    auto filter_png = Gtk::FileFilter::create();
    filter_png->set_name("PNG files");
    filter_png->add_mime_type("image/png");
    filters->append(filter_png);

    auto filter_any = Gtk::FileFilter::create();
    filter_any->set_name("Any files");
    filter_any->add_pattern("*");
    filters->append(filter_any);

    dialog->set_filters(filters);

    dialog->open(sigc::bind(sigc::mem_fun(*this, &MainWindow::onLoadImageDialogFinish),
                            dialog));
}

void MainWindow::onLoadImageDialogFinish(const Glib::RefPtr<Gio::AsyncResult>& result,
                                         const Glib::RefPtr<Gtk::FileDialog>& dialog)
{
    try {
        auto file = dialog->open_finish(result);

        auto filename = file->get_path();
        fmt::println("Loading image: {}", filename);
    } catch (const Gtk::DialogError& err) {
        fmt::println("No image loaded: {}", err.what());
    }
}
