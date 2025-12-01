# Copyright (c) 2023 Airbyte, Inc., all rights reserved.
from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

from airbyte import exceptions as exc
from airbyte._executors.base import Executor
from airbyte._message_iterators import AirbyteMessageIterator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from airbyte._executors.base import IO


logger = logging.getLogger("airbyte")


DEFAULT_AIRBYTE_CONTAINER_TEMP_DIR = "/airbyte/tmp"
"""Default temp dir in an Airbyte connector's Docker image."""


class DockerExecutor(Executor):
    def __init__(
        self,
        name: str,
        image_name_full: str,
        *,
        executable: list[str],
        target_version: str | None = None,
        volumes: dict[Path, str] | None = None,
    ) -> None:
        self.executable: list[str] = executable
        self.volumes: dict[Path, str] = volumes or {}
        self.image_name_full: str = image_name_full
        self._docker_client: docker.DockerClient | None = None
        super().__init__(name=name, target_version=target_version)

    @property
    def docker_client(self) -> docker.DockerClient:
        """Get or create the Docker client."""
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except DockerException as e:
                raise exc.AirbyteConnectorExecutableNotFoundError(
                    connector_name=self.name,
                ) from e
        return self._docker_client

    def _parse_docker_run_args(self) -> dict:
        """Parse the executable list to extract Docker run parameters."""
        # The executable format is: ["docker", "run", "--rm", "-i", "--volume", ...]
        # We need to extract volumes, network, and other options
        volumes: dict[str, dict[str, str]] = {}
        network_mode: str | None = None
        remove: bool = False
        stdin_open: bool = False

        i = 0
        while i < len(self.executable):
            arg = self.executable[i]
            if arg == "--volume" and i + 1 < len(self.executable):
                volume_spec = self.executable[i + 1]
                # Format: "local_path:container_path"
                if ":" in volume_spec:
                    local_path, container_path = volume_spec.rsplit(":", 1)
                    volumes[local_path] = {"bind": container_path, "mode": "rw"}
                i += 2
            elif arg == "--network" and i + 1 < len(self.executable):
                network_mode = self.executable[i + 1]
                i += 2
            elif arg == "--rm":
                remove = True
                i += 1
            elif arg == "-i":
                stdin_open = True
                i += 1
            else:
                i += 1

        return {
            "volumes": volumes,
            "network_mode": network_mode,
            "remove": remove,
            "stdin_open": stdin_open,
        }

    def ensure_installation(
        self,
        *,
        auto_fix: bool = True,
    ) -> None:
        """Ensure that the connector executable can be found.

        The auto_fix parameter is ignored for this executor type.
        """
        _ = auto_fix
        try:
            # Check if Docker is available
            self.docker_client.ping()
            # Test by running spec command
            list(self.execute(["spec"]))
        except Exception as e:
            raise exc.AirbyteConnectorExecutableNotFoundError(
                connector_name=self.name,
            ) from e

    def install(self) -> None:
        """Install the connector.

        For docker images, for now this is a no-op. In the future we might
        pull the Docker image in this step.
        """
        pass

    def uninstall(self) -> None:
        """Uninstall the connector.

        For docker images, this operation removes the Docker image using the Docker Python library.

        We suppress any errors that occur during the removal process.
        """
        with suppress(ImageNotFound, NotFound, DockerException):
            self.docker_client.images.remove(self.image_name_full)

    @property
    def _cli(self) -> list[str]:
        """Get the base args of the CLI executable."""
        return self.executable

    def execute(
        self,
        args: list[str],
        *,
        stdin: IO[str] | AirbyteMessageIterator | None = None,
        suppress_stderr: bool = False,
    ) -> Iterator[str]:
        """Execute a command in a Docker container and return an iterator of STDOUT lines.

        If stdin is provided, it will be passed to the container as STDIN.
        If suppress_stderr is True, stderr output will be suppressed to reduce noise.
        """
        import threading

        mapped_args = self.map_cli_args(args)
        run_params = self._parse_docker_run_args()

        # Prepare command - the image name is the last element in executable before args
        image_name = self.image_name_full
        command = mapped_args

        # Determine if we need stdin
        needs_stdin = stdin is not None
        stdin_open = run_params["stdin_open"] or needs_stdin

        try:
            # Create and run the container
            container = self.docker_client.containers.run(
                image_name,
                command=command,
                volumes=run_params["volumes"],
                network_mode=run_params["network_mode"],
                remove=run_params["remove"],
                stdin_open=stdin_open,
                detach=True,
                stdout=True,
                stderr=not suppress_stderr,
            )

            # Handle stdin in a separate thread if needed
            stdin_thread: threading.Thread | None = None
            if stdin is not None:
                if isinstance(stdin, AirbyteMessageIterator):
                    # Create a thread to pump messages into stdin
                    def pump_stdin():
                        try:
                            socket = container.attach_socket(
                                params={"stdin": 1, "stdout": 0, "stderr": 0}
                            )
                            try:
                                for message in stdin:
                                    if not container.attrs.get("State", {}).get("Running", False):
                                        break
                                    socket._sock.send(
                                        (message.model_dump_json() + "\n").encode("utf-8")
                                    )
                                socket._sock.shutdown(2)  # SHUT_RDWR
                            except (BrokenPipeError, OSError):
                                pass  # Expected during graceful shutdown
                            finally:
                                try:
                                    socket.close()
                                except Exception:
                                    pass
                        except (BrokenPipeError, OSError):
                            pass  # Expected during graceful shutdown
                        except Exception:
                            pass  # Suppress other errors

                    stdin_thread = threading.Thread(target=pump_stdin, daemon=True)
                    stdin_thread.start()
                else:
                    # Handle file-like stdin
                    def pump_stdin():
                        try:
                            socket = container.attach_socket(
                                params={"stdin": 1, "stdout": 0, "stderr": 0}
                            )
                            try:
                                if hasattr(stdin, "read"):
                                    data = stdin.read()
                                    if data:
                                        socket._sock.send(data.encode("utf-8"))
                                socket._sock.shutdown(2)  # SHUT_RDWR
                            except (BrokenPipeError, OSError):
                                pass
                            finally:
                                try:
                                    socket.close()
                                except Exception:
                                    pass
                        except (BrokenPipeError, OSError):
                            pass
                        except Exception:
                            pass

                    stdin_thread = threading.Thread(target=pump_stdin, daemon=True)
                    stdin_thread.start()

            # Stream output line by line
            try:
                for line in container.logs(stream=True, follow=True, stdout=True, stderr=False):
                    decoded_line = line.decode("utf-8")
                    # Remove trailing newlines but preserve the line content
                    if decoded_line.endswith("\n"):
                        decoded_line = decoded_line[:-1]
                    if decoded_line.endswith("\r"):
                        decoded_line = decoded_line[:-1]
                    yield decoded_line
            finally:
                # Wait for container to finish
                exit_code = container.wait()["StatusCode"]

                # Wait for stdin thread to finish if it exists
                if stdin_thread is not None:
                    stdin_thread.join(timeout=1.0)

                # Clean up if not auto-removed
                if not run_params["remove"]:
                    try:
                        container.remove()
                    except NotFound:
                        pass  # Container already removed

                # Raise exception if container exited with non-zero code
                if exit_code not in {0, -15}:  # 0 = success, -15 = SIGTERM
                    raise exc.AirbyteSubprocessFailedError(
                        run_args=[*self._cli, *mapped_args],
                        exit_code=exit_code,
                        original_exception=None,
                    )

        except ImageNotFound as e:
            raise exc.AirbyteConnectorExecutableNotFoundError(
                connector_name=self.name,
            ) from e
        except DockerException as e:
            raise exc.AirbyteSubprocessError(
                message=f"Docker execution failed: {str(e)}",
                context={
                    "args": mapped_args,
                    "image": image_name,
                },
            ) from e

    def map_cli_args(self, args: list[str]) -> list[str]:
        """Map local file paths to the container's volume paths."""
        new_args = []
        for arg in args:
            if Path(arg).exists():
                # This is a file path and we need to map it to the same file within the
                # relative path of the file within the container's volume.
                for local_volume, container_path in self.volumes.items():
                    if Path(arg).is_relative_to(local_volume):
                        logger.debug(
                            f"Found file input path `{arg}` "
                            f"relative to container-mapped volume: {local_volume}"
                        )
                        mapped_path = Path(container_path) / Path(arg).relative_to(local_volume)
                        logger.debug(f"Mapping `{arg}` -> `{mapped_path}`")
                        new_args.append(str(mapped_path))
                        break
                else:
                    # No break reached; a volume was found for this file path
                    logger.warning(
                        f"File path `{arg}` is not relative to any volume path "
                        f"in the provided volume mappings: {self.volumes}. "
                        "The file may not be available to the container at runtime."
                    )
                    new_args.append(arg)

            else:
                new_args.append(arg)

        if args != new_args:
            logger.debug(
                f"Mapping local-to-container CLI args: {args} -> {new_args} "
                f"based upon volume definitions: {self.volumes}"
            )

        return new_args
