#!/usr/local/bin/env python3

# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import socket
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM


class VsockStream:
    """Client"""
    def __init__(self, conn_tmo=5):
        self.conn_tmo = conn_tmo

    def connect(self, endpoint):
        """Connect to the remote endpoint"""
        self.sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        self.sock.settimeout(self.conn_tmo)
        self.sock.connect(endpoint)

    def send_data(self, data):
        """Send data to a remote endpoint"""
        self.sock.sendall(data)
        self.sock.shutdown(socket.SHUT_WR)

    def recv_data(self):
        """Receive data from a remote endpoint"""
        while True:
            data = self.sock.recv(1024).decode()
            if not data:
                break
            print(data, end='', flush=True)
        print()

    def disconnect(self):
        """Close the client socket"""
        self.sock.close()


def client_handler(args):
    client = VsockStream()
    endpoint = (args.cid, args.port)
    client.connect(endpoint)
    msg = args.prompt if hasattr(args, 'prompt') else 'Hello, world!'
    client.send_data(msg.encode())
    client.recv_data()
    client.disconnect()


class VsockListener:
    """Server"""
    def __init__(self, conn_backlog=128):
        self.conn_backlog = conn_backlog
        self.model = None
        self.tokenizer = None

    def bind(self, port):
        """Bind and listen for connections on the specified port"""
        self.sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        self.sock.bind((socket.VMADDR_CID_ANY, port))
        self.sock.listen(self.conn_backlog)

    def recv_data(self):
        """Receive data from a remote endpoint"""
        while True:
            (from_client, (remote_cid, remote_port)) = self.sock.accept()
            # Read all data
            prompt = ""
            try:
                while True:
                    data = from_client.recv(1024).decode()
                    if not data:
                        break
                    prompt += data
            except socket.error:
                pass
            
            if prompt and self.model and self.tokenizer:
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_length=100)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                from_client.sendall(response.encode())
            from_client.close()

    def send_data(self, data):
        """Send data to a renote endpoint"""
        while True:
            (to_client, (remote_cid, remote_port)) = self.sock.accept()
            to_client.sendall(data)
            to_client.close()


def server_handler(args):
    server = VsockListener()
    # Load model
    model_path = "enclave/bloom"
    print(f"Loading model from {model_path}...", flush=True)
    try:
        if not os.path.exists(model_path):
            print(f"ERROR: Model path {model_path} does not exist!", flush=True)
            sys.exit(1)
        server.tokenizer = AutoTokenizer.from_pretrained(model_path)
        server.model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    server.bind(args.port)
    print(f"Server listening on port {args.port}", flush=True)
    server.recv_data()


def main():
    parser = argparse.ArgumentParser(prog='vsock-sample')
    parser.add_argument("--version", action="version",
                        help="Prints version information.",
                        version='%(prog)s 0.1.0')
    subparsers = parser.add_subparsers(title="options")

    client_parser = subparsers.add_parser("client", description="Client",
                                          help="Connect to a given cid and port.")
    client_parser.add_argument("cid", type=int, help="The remote endpoint CID.")
    client_parser.add_argument("port", type=int, help="The remote endpoint port.")
    client_parser.add_argument("--prompt", type=str, help="Prompt to send to the LLM.")
    client_parser.set_defaults(func=client_handler)

    server_parser = subparsers.add_parser("server", description="Server",
                                          help="Listen on a given port.")
    server_parser.add_argument("port", type=int, help="The local port to listen on.")
    server_parser.set_defaults(func=server_handler)

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
