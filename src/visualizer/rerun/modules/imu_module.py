import rerun as rr

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class ImuModule(IVizModule):
    """Imu module."""

    def __init__(
        self,
        entity_path: str,
        fields: list[str],
        red_curve_color: list[int] | None = None,
        green_curve_color: list[int] | None = None,
        blue_curve_color: list[int] | None = None,
    ) -> None:
        """Initialize the imu module."""
        self.entity_path = entity_path
        self.fields = fields
        self.logger = spawn_logger(ImuModule.__name__)
        self.static_pushed = False
        self.red_curve_color = red_curve_color or [255, 60, 60]
        self.green_curve_color = green_curve_color or [60, 255, 160]
        self.blue_curve_color = blue_curve_color or [30, 210, 255]

    def setup(self) -> None:
        """Set up the image module."""

    def process(self, context: Ctx) -> None:
        """Process the image data."""
        for field in self.fields:
            exists = context.exists(field)
            if not exists:
                msg = f"Imu data not found in context: {field}"
                self.logger.warning(msg)
                raise KeyError(msg)

        pairs = []
        imu_rows = context.get_scalar("imu_rows", int)
        imu_ts = context.get_ndarray("imu_ts", (imu_rows,))
        imu_ts = imu_ts / 1e9
        for field in self.fields:
            columns = context.get_ndarray(field, (imu_rows, 3))
            path = f"{self.entity_path}/{field}"
            pairs.append((path, columns, field))

        if not self.static_pushed:
            for path, _, field in pairs:
                x_path = f"{path}/x"
                y_path = f"{path}/y"
                z_path = f"{path}/z"
                rr.log(
                    x_path,
                    rr.SeriesLines(colors=self.red_curve_color, names=[f"{field}_x"], widths=[1]),
                    static=True,
                )
                rr.log(
                    y_path,
                    rr.SeriesLines(colors=self.green_curve_color, names=[f"{field}_y"], widths=[1]),
                    static=True,
                )
                rr.log(
                    z_path,
                    rr.SeriesLines(colors=self.blue_curve_color, names=[f"{field}_z"], widths=[1]),
                    static=True,
                )
            self.static_pushed = True
        for path, value, _ in pairs:
            rr.send_columns(
                f"{path}/x",
                indexes=[rr.TimeColumn("sim_time", timestamp=imu_ts)],
                columns=rr.Scalars.columns(scalars=value[:, 0]),
            )
            rr.send_columns(
                f"{path}/y",
                indexes=[rr.TimeColumn("sim_time", timestamp=imu_ts)],
                columns=rr.Scalars.columns(scalars=value[:, 1]),
            )
            rr.send_columns(
                f"{path}/z",
                indexes=[rr.TimeColumn("sim_time", timestamp=imu_ts)],
                columns=rr.Scalars.columns(scalars=value[:, 2]),
            )

    def __repr__(self) -> str:
        """Return the string representation of the image module."""
        return f"ImuModule(entity_path={self.entity_path}, fields={self.fields})"
