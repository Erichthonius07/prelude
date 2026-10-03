package com.prelude.denoise.model;

/**
 * Local value object carrying the image dimensions that FusedFrame does not have.
 * Supplied explicitly by the caller (contract gap: §6.3 defines no width/height/channels).
 * See .agents/5a/OPEN_QUESTIONS.md.
 */
public final class FrameGeometry {
    private final int width;
    private final int height;
    private final int channels;

    public FrameGeometry(int width, int height, int channels) {
        if (width <= 0 || height <= 0 || channels <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    public int getWidth()    { return width; }
    public int getHeight()   { return height; }
    public int getChannels() { return channels; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FrameGeometry)) return false;
        FrameGeometry that = (FrameGeometry) o;
        return width == that.width && height == that.height && channels == that.channels;
    }

    @Override
    public int hashCode() {
        return 31 * (31 * width + height) + channels;
    }
}
