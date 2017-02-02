package vax.snnt.neuronet;

import java.util.Random;

/**

 @author toor
 */
public class Rng {
    public static final Random rng = new Random( /*1410*/ );
    public static double rndMax = 3.5; // 2.5 // 3.5

    public static double rnd () {
        return ( rng.nextDouble() - 0.5 ) * 2 * rndMax;
    }

    public static double nextDouble () {
        return rng.nextDouble();
    }

    public static int nextInt ( int bound ) {
        return rng.nextInt( bound );
    }
}
