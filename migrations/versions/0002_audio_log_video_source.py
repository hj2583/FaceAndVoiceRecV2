"""Associate uploaded-video audio logs with their source video."""

from alembic import op
import sqlalchemy as sa


revision = "0002_audio_log_video_source"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audio_logs", sa.Column("source_ref", sa.Text()))


def downgrade() -> None:
    op.drop_column("audio_logs", "source_ref")