"""Add capability bindings (skills, tools, MCP servers) to applications.

Revision ID: 019
Revises: 018
Create Date: 2026-05-24 12:00:00

Adds:
- Tenant-level default capability name arrays on `applications`
  (default_skill_names, default_mcp_server_names, default_tool_names).
- Workspace-level tool override columns on `workspace_applications`
  (tool_names) plus per-kind override modes (skill/mcp/tool).
- Two new junction tables for workspace-level skill and MCP-server
  overrides keyed by (workspace_id, application_id):
  `workspace_application_skills` and `workspace_application_mcp_servers`.

Skills and MCP servers are workspace-scoped resources (FK to workspaces.id),
so tenant defaults are stored as names (resolved per workspace at load),
while workspace overrides hold real FKs to the workspace's own rows.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply capability binding schema."""
    # ---- applications: tenant-level defaults ----
    op.add_column(
        "applications",
        sa.Column(
            "default_skill_names",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "applications",
        sa.Column(
            "default_mcp_server_names",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "applications",
        sa.Column(
            "default_tool_names",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )

    # ---- workspace_applications: per-binding overrides ----
    op.add_column(
        "workspace_applications",
        sa.Column(
            "tool_names",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )
    op.add_column(
        "workspace_applications",
        sa.Column(
            "skill_override_mode",
            sa.Text(),
            nullable=False,
            server_default="merge",
        ),
    )
    op.add_column(
        "workspace_applications",
        sa.Column(
            "mcp_override_mode",
            sa.Text(),
            nullable=False,
            server_default="merge",
        ),
    )
    op.add_column(
        "workspace_applications",
        sa.Column(
            "tool_override_mode",
            sa.Text(),
            nullable=False,
            server_default="merge",
        ),
    )

    # workspace_applications must have a composite UNIQUE so the
    # workspace_application_* junctions can reference it. The existing
    # PRIMARY KEY (workspace_id, application_id) already serves; no extra
    # constraint required for the composite FK below.

    # ---- workspace_application_skills junction ----
    op.create_table(
        "workspace_application_skills",
        sa.Column("workspace_id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("application_id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "skill_id",
            sa.Text(),
            sa.ForeignKey("skills.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "application_id"],
            ["workspace_applications.workspace_id", "workspace_applications.application_id"],
            ondelete="CASCADE",
            name="fk_wa_skills_binding",
        ),
    )
    op.create_index(
        "idx_wa_skills_binding",
        "workspace_application_skills",
        ["workspace_id", "application_id"],
    )
    op.create_index(
        "idx_wa_skills_skill",
        "workspace_application_skills",
        ["skill_id"],
    )

    # ---- workspace_application_mcp_servers junction ----
    op.create_table(
        "workspace_application_mcp_servers",
        sa.Column("workspace_id", sa.Text(), nullable=False, primary_key=True),
        sa.Column("application_id", sa.Text(), nullable=False, primary_key=True),
        sa.Column(
            "mcp_server_id",
            sa.Text(),
            sa.ForeignKey("mcp_servers.id", ondelete="CASCADE"),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "application_id"],
            ["workspace_applications.workspace_id", "workspace_applications.application_id"],
            ondelete="CASCADE",
            name="fk_wa_mcp_binding",
        ),
    )
    op.create_index(
        "idx_wa_mcp_binding",
        "workspace_application_mcp_servers",
        ["workspace_id", "application_id"],
    )
    op.create_index(
        "idx_wa_mcp_server",
        "workspace_application_mcp_servers",
        ["mcp_server_id"],
    )


def downgrade() -> None:
    """Drop capability binding schema."""
    op.drop_index("idx_wa_mcp_server", table_name="workspace_application_mcp_servers")
    op.drop_index("idx_wa_mcp_binding", table_name="workspace_application_mcp_servers")
    op.drop_table("workspace_application_mcp_servers")

    op.drop_index("idx_wa_skills_skill", table_name="workspace_application_skills")
    op.drop_index("idx_wa_skills_binding", table_name="workspace_application_skills")
    op.drop_table("workspace_application_skills")

    op.drop_column("workspace_applications", "tool_override_mode")
    op.drop_column("workspace_applications", "mcp_override_mode")
    op.drop_column("workspace_applications", "skill_override_mode")
    op.drop_column("workspace_applications", "tool_names")

    op.drop_column("applications", "default_tool_names")
    op.drop_column("applications", "default_mcp_server_names")
    op.drop_column("applications", "default_skill_names")
