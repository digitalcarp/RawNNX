// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2025 Daniel Gao

#include "mainwindow.h"

#include <gtkmm/application.h>

int main(int argc, char** argv)
{
    auto app = Gtk::Application::create("org.rawnnx.sandbox");

    return app->make_window_and_run<MainWindow>(argc, argv);
}
