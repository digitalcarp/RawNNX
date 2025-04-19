// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#pragma once

#include <gtkmm/window.h>

class ImageArea;

namespace Gio {
class AsyncResult;
}
namespace Gtk {
class FileDialog;
}

class MainWindow : public Gtk::Window {
public:
    MainWindow();

private:
    void onLoadImageButtonClicked();
    void onLoadImageDialogFinish(const Glib::RefPtr<Gio::AsyncResult>& result,
                                 const Glib::RefPtr<Gtk::FileDialog>& dialog);

    ImageArea* m_img_area;
};
