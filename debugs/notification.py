import gi

gi.require_version('Gio', '2.0')
from gi.repository import Gio

Application = Gio.Application.new("hello.world", Gio.ApplicationFlags.FLAGS_NONE)
Application.register()
Notification = Gio.Notification.new("Hello world")
Notification.set_body("This is an example notification.")
Icon = Gio.ThemedIcon.new("dialog-information")
Notification.set_icon(Icon)
Application.send_notification(None, Notification)
