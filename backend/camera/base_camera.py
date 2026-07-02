from abc import ABC, abstractmethod


class BaseCamera(ABC):

    def __init__(self, camera_id: str, zone: str):

        self.camera_id = camera_id
        self.zone = zone

    @abstractmethod
    def update(self):
        """
        Update camera state.
        """
        pass

    @abstractmethod
    def get_dashboard_data(self):
        """
        Return JSON-ready dashboard payload.
        """
        pass