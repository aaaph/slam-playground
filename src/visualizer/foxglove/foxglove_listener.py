import threading

from foxglove.websocket import (
    ChannelView,
    Client,
    ClientChannel,
    ServerListener,
)

from logger import spawn_logger


class FoxgloveServerListener(ServerListener):
    """Foxglove server listener."""

    def __init__(self) -> None:
        """Initialize the Foxglove server listener."""
        super().__init__()
        self.subscribers: dict[int, set[str]] = {}
        self.client_connected_event = threading.Event()
        self.logger = spawn_logger(app="foxglove_server_listener")

    def has_subscribers(self) -> bool:
        """Check if there are any subscribers."""
        return len(self.subscribers) > 0

    def on_subscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """
        Handle client subscription to a channel.

        We'll use this and on_unsubscribe to simply track if we have any subscribers at all.
        """
        self.logger.info(f"Client {client} subscribed to channel {channel.topic}")
        self.subscribers.setdefault(client.id, set()).add(channel.topic)
        self.client_connected_event.set()

    def on_unsubscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """Handle client unsubscription from a channel."""
        self.logger.info(f"Client {client} unsubscribed from channel {channel.topic}")
        self.subscribers[client.id].remove(channel.topic)
        if not self.subscribers[client.id]:
            del self.subscribers[client.id]
            if not self.subscribers:
                self.client_connected_event.clear()
                self.logger.info("No subscribers left, clearing client connected event")

    def on_client_advertise(
        self,
        client: Client,
        channel: ClientChannel,
    ) -> None:
        """Handle client advertisement of a new channel."""
        self.logger.info(f"Client {client.id} advertised channel: {channel.id}")
        self.logger.info(f"  Topic: {channel.topic}")
        self.logger.info(f"  Encoding: {channel.encoding}")
        self.logger.info(f"  Schema name: {channel.schema_name}")
        self.logger.info(f"  Schema encoding: {channel.schema_encoding}")
        self.logger.info(f"  Schema: {channel.schema!r}")

    def on_message_data(
        self,
        client: Client,
        client_channel_id: int,
        data: bytes,
    ) -> None:
        """
        Handle receiving messages from the client.

        You can send messages from Foxglove app in the publish panel:
        https://docs.foxglove.dev/docs/visualization/panels/publish
        """
        self.logger.info(f"Message from client {client.id} on channel {client_channel_id}")
        self.logger.info(f"Data: {data!r}")

    def on_client_unadvertise(
        self,
        client: Client,
        client_channel_id: int,
    ) -> None:
        """Handle client unadvertisement of a channel."""
        self.logger.info(f"Client {client.id} unadvertised channel: {client_channel_id}")
