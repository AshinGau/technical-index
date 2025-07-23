#!/usr/bin/env python3
"""
Main entry point for the technical-index application.
"""

import warnings

# 禁止pkg_resources相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


def main():
    """Main function that prints Hello World."""
    print("Hello World!")


if __name__ == "__main__":
    main()
