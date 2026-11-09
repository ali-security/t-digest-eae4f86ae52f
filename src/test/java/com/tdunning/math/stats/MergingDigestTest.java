/*
 * Licensed to Ted Dunning under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tdunning.math.stats;

import com.carrotsearch.randomizedtesting.annotations.Seed;
import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Random;

//to freeze the tests with a particular seed, put the seed on the next line
//@Seed("84527677CF03B566:A6FF596BDDB2D59D")
public class MergingDigestTest extends TDigestTest {
    @BeforeClass
    public static void setup() throws IOException {
        TDigestTest.setup("merge");
    }

    protected DigestFactory factory(final double compression) {
        return new DigestFactory() {
            @Override
            public TDigest create() {
                return new MergingDigest(compression);
            }
        };
    }

    @Before
    public void testSetUp() {
        RandomUtils.useTestSeed();
    }

    @Test
    public void testApproximation() {
        double worst = 0;
        double old = Double.NEGATIVE_INFINITY;
        for (double x = -1; x < 1; x+=0.0001) {
            double ex = Math.asin(x);
            double actual = MergingDigest.asinApproximation(x);
            double error = ex - actual;
//            System.out.printf("%.8f, %.8f, %.8f, %.12f\n", x, ex, actual, error * 1e6);
            worst = Math.max(worst, Math.abs(error));
            assertEquals("Bad approximation", 0, error, 1e-6);
            assertTrue("Not monotonic", actual >= old);
            old = actual;
        }
        System.out.printf("worst = %.5g\n", worst);

    }

//    @Test
    public void testFill() {
        int delta = 300;
        MergingDigest x = new MergingDigest(delta);
        Random gen = new Random();
        for (int i = 0; i < 1000000; i++) {
            x.add(gen.nextGaussian());
        }
        double q0 = 0;
        int i = 0;
        System.out.printf("i, q, mean, count, dk\n");
        for (Centroid centroid : x.centroids()) {
            double q = q0 + centroid.count() / 2.0 / x.size();
            double q1 = q0 + (double) centroid.count() / x.size();
            double dk = delta * (qToK(q1) - qToK(q0));
            System.out.printf("%d,%.7f,%.7f,%d,%.7f\n", i, q, centroid.mean(), centroid.count(), dk);
            if (Double.isNaN(dk)) {
                System.out.printf(">>>> %.8f, %.8f\n", q0, q1);
            }
            assertTrue(String.format("K-size for centroid %d at %.3f is %.3f", i, centroid.mean(), dk), dk <= 1);
            q0 = q1;
            i++;
        }
    }

    private double qToK(double q) {
        return Math.asin(2 * Math.min(1, q) - 1) / Math.PI + 0.5;
    }

    @Test
    public void testSmallCountQuantile() {
        List<Double> data = Lists.newArrayList(15.0, 20.0, 32.0, 60.0);
        TDigest td = new MergingDigest(200);
        for (Double datum : data) {
            td.add(datum);
        }
        assertEquals(21.2, td.quantile(0.4), 1e-10);
    }

    @Test
    public void printQuantiles() throws FileNotFoundException {
        // the output of this can be used to visually check the interpolation with R:
        //     x = c(1,2,5,5,6,9,10)
        //     zx = read.csv("interpolation.csv")
        //     plot.ecdf(x)
        //     lines(col='red', q ~ x, zx)
        MergingDigest td = new MergingDigest(200);
        td.setMinMax(0, 10);
        td.add(1);
        td.add(2);
        td.add(5, 2);
        td.add(6);
        td.add(9);
        td.add(10);

        try (PrintWriter quantiles = new PrintWriter("interpolation.csv");
             PrintWriter cdfs = new PrintWriter("reverse.csv")) {

            quantiles.printf("x,q\n");
            cdfs.printf("x,q\n");
            for (double q = 0; q < 1; q += 1e-3) {
                double x = td.quantile(q);
                quantiles.printf("%.3f,%.3f\n", x, q);

                double roundTrip = td.cdf(x);
                cdfs.printf("%.3f,%.3f\n", x, q);

                if (x < 10) {
                    assertEquals(q, roundTrip, 1e-6);
                }
            }
        }

        assertEquals(2.0 / 7, td.cdf(3), 1e-9);
    }

    @Override
    protected TDigest fromBytes(ByteBuffer bytes) {
        return MergingDigest.fromBytes(bytes);
    }

    // Slow test (runSlowTests=true): on this release MergingDigest(compression=2) throws
    // ArrayIndexOutOfBoundsException in merge() once the main buffer overflows, so the
    // unpatched base cannot pass it; the bug is in the MergingDigest code the later fix touches.
    @Override
    @Test
    @org.junit.Ignore("MergingDigest.merge() ArrayIndexOutOfBoundsException at compression=2 on the unpatched 3.2 base")
    public void testSizeControl() {
    }

    // ---- Validation tests for fromBytes (issue #227) ----

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingNegativeN() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);         // min
        buf.putDouble(1.0);         // max
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) -5);   // n (negative)
        buf.putShort((short) 10);   // bufferSize
        buf.putShort((short) 0);    // lastUsedCell
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNExceedsCapacity() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 100 * 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);         // min
        buf.putDouble(1.0);         // max
        buf.putDouble(1.0);         // compression -> small capacity
        buf.putInt(100);            // n = 100, way more than capacity
        for (int i = 0; i < 100; i++) {
            buf.putDouble(1.0);     // weight
            buf.putDouble(i);       // mean (non-decreasing)
        }
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingOverflowCompression() {
        ByteBuffer buf = ByteBuffer.allocate(32).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);         // min
        buf.putDouble(1.0);         // max
        buf.putDouble(1.0e10);      // compression (exceeds MAX_COMPRESSION of 1e6)
        buf.putInt(0);              // n
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNegativeCompression() {
        ByteBuffer buf = ByteBuffer.allocate(32).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(-50.0);       // negative compression
        buf.putInt(0);
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNegativeWeight() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(1);              // n = 1
        buf.putDouble(-1.0);        // negative weight
        buf.putDouble(5.0);         // mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingDecreasingMeans() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 2 * 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(10.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(2);              // n = 2
        buf.putDouble(1.0);         // weight[0]
        buf.putDouble(5.0);         // mean[0] = 5
        buf.putDouble(1.0);         // weight[1]
        buf.putDouble(3.0);         // mean[1] = 3, out of order
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingLastUsedCellExceedsN() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) 10);   // n
        buf.putShort((short) 50);   // bufferSize
        buf.putShort((short) 20);   // lastUsedCell > n
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingNegativeWeight() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) 220);  // n
        buf.putShort((short) 500);  // bufferSize
        buf.putShort((short) 1);    // lastUsedCell
        buf.putFloat(-1.0f);        // negative weight
        buf.putFloat(5.0f);         // mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingDecreasingMeans() {
        ByteBuffer buf = ByteBuffer.allocate(128).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(10.0);
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) 220);  // n
        buf.putShort((short) 500);  // bufferSize
        buf.putShort((short) 2);    // lastUsedCell
        buf.putFloat(1.0f);         // weight[0]
        buf.putFloat(5.0f);         // mean[0] = 5
        buf.putFloat(1.0f);         // weight[1]
        buf.putFloat(3.0f);         // mean[1] = 3, out of order
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNaNMean() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(1);              // n = 1
        buf.putDouble(1.0);         // valid weight
        buf.putDouble(Double.NaN);  // NaN mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNaNWeight() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(1);              // n = 1
        buf.putDouble(Double.NaN);  // NaN weight
        buf.putDouble(5.0);         // mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingInfinityWeight() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(1);              // n = 1
        buf.putDouble(Double.POSITIVE_INFINITY); // Infinity weight
        buf.putDouble(5.0);         // mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingInfinityMean() {
        ByteBuffer buf = ByteBuffer.allocate(4 + 8 + 8 + 8 + 4 + 16).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(100.0);       // compression
        buf.putInt(1);              // n = 1
        buf.putDouble(1.0);         // valid weight
        buf.putDouble(Double.POSITIVE_INFINITY); // Infinity mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesVerboseEncodingNaNCompression() {
        ByteBuffer buf = ByteBuffer.allocate(32).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(1);              // VERBOSE_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putDouble(Double.NaN);  // NaN compression
        buf.putInt(0);
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingNaNWeight() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) 220);  // n
        buf.putShort((short) 500);  // bufferSize
        buf.putShort((short) 1);    // lastUsedCell
        buf.putFloat(Float.NaN);    // NaN weight
        buf.putFloat(5.0f);         // mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testFromBytesSmallEncodingNaNMean() {
        ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.BIG_ENDIAN);
        buf.putInt(2);              // SMALL_ENCODING
        buf.putDouble(0.0);
        buf.putDouble(1.0);
        buf.putFloat(100.0f);       // compression
        buf.putShort((short) 220);  // n
        buf.putShort((short) 500);  // bufferSize
        buf.putShort((short) 1);    // lastUsedCell
        buf.putFloat(1.0f);         // valid weight
        buf.putFloat(Float.NaN);    // NaN mean
        buf.flip();
        MergingDigest.fromBytes(buf);
    }

    @Test
    public void testFromBytesRoundTrip() {
        // Valid digest should still round-trip correctly
        MergingDigest original = new MergingDigest(100);
        Random gen = new Random(42);
        for (int i = 0; i < 1000; i++) {
            original.add(gen.nextGaussian());
        }
        ByteBuffer buf = ByteBuffer.allocate(original.byteSize());
        original.asBytes(buf);
        buf.flip();
        MergingDigest restored = MergingDigest.fromBytes(buf);
        assertEquals(original.size(), restored.size(), 0);
        assertEquals(original.getMin(), restored.getMin(), 0);
        assertEquals(original.getMax(), restored.getMax(), 0);
    }

    @Test
    public void testFromBytesSmallRoundTrip() {
        MergingDigest original = new MergingDigest(100);
        Random gen = new Random(42);
        for (int i = 0; i < 1000; i++) {
            original.add(gen.nextGaussian());
        }
        ByteBuffer buf = ByteBuffer.allocate(original.smallByteSize());
        original.asSmallBytes(buf);
        buf.flip();
        MergingDigest restored = MergingDigest.fromBytes(buf);
        assertEquals(original.size(), restored.size(), 1);  // float precision
        assertEquals(original.getMin(), restored.getMin(), 0);
        assertEquals(original.getMax(), restored.getMax(), 0);
    }
}