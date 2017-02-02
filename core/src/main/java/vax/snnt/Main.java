package vax.snnt;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import vax.snnt.neuronet.*;
import static vax.snnt.neuronet.Misc.*;
import static vax.snnt.neuronet.Rng.rnd;
import static vax.snnt.neuronet.TransferFunction.*;

/**

 @author toor
 */
public class Main {

    /**
     @param args the command line arguments
     */
    public static void main ( String[] args ) {
        test6();
    }

    // 0-layer network
    public static void test1 () {
        Network network = new Network();

        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ) );

        network.addOutputConsumer( outN, System.out::println );

        Layer inputLayer = new Layer();
        Layer outputLayer = new Layer();

        network.addLayer( inputLayer );
        network.addLayer( outputLayer );

        int inNr = network.addNeuron( inN, inputLayer );
        int outNr = network.addNeuron( outN, outputLayer );
        network.addConnection( inNr, outNr, 2.0 );
        inN.add( 0.42 );

        network.process();

        // 0.84
    }

    // 1-layer network
    public static void test2 () {
        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ) );

        Layer inputLayer = new Layer();
        Layer midLayer = new Layer();
        Layer outputLayer = new Layer();

        Network network = new Network();
        network.addLayer( inputLayer );
        network.addLayer( midLayer );
        network.addLayer( outputLayer );

        network.addOutputConsumer( outN, System.out::println );

        int inNr = network.addNeuron( inN, inputLayer );
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            network.addNeuron( neuron, midLayer );
        }
        int outNr = network.addNeuron( outN, outputLayer );

        for( Neuron neuron : network.layerIterable( midLayer ) ) {
            network.addConnection( inN, neuron, 1.0 );
            network.addConnection( neuron, outN, 0.05 );
        }

        inN.add( 0.42 );
        network.process();
        // v = 0.20999999999999996
    }

    // 2-layer network
    public static void test3 () {
        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ) );

        Layer inputLayer = new Layer();
        Layer midLayer1 = new Layer();
        Layer midLayer2 = new Layer();
        Layer outputLayer = new Layer();

        Network network = new Network();
        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        network.addOutputConsumer( outN, System.out::println );

        network.addNeuron( inN, inputLayer );
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            network.addNeuron( neuron, midLayer1 );
        }
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            network.addNeuron( neuron, midLayer2 );
        }
        network.addNeuron( outN, outputLayer );

        for( Neuron neuron : network.layerIterable( midLayer1 ) ) {
            network.addConnection( inN, neuron, 1.0 );
            for( Neuron neuron2 : network.layerIterable( midLayer2 ) ) {
                network.addConnection( neuron, neuron2, 0.1 );
            }
        }

        for( Neuron neuron : network.layerIterable( midLayer2 ) ) {
            network.addConnection( neuron, outN, 0.05 );
        }

        inN.add( 0.42 );
        network.process();
        // v = 0.20999999999999996
    }

    private static Network createTestTwoLayerNetwork1 () {
        Network network = new Network();

        Neuron inN = new Neuron( IDENTITY, IDENTITY );
        Neuron outN = new Neuron( IDENTITY, IDENTITY );

        int neuronsInLayer = 10;

        Layer inputLayer = new Layer();
        Layer midLayer1 = new Layer();
        Layer midLayer2 = new Layer();
        Layer outputLayer = new Layer();

        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        network.addNeuron( inN, inputLayer );
        for( int i = 0; i < neuronsInLayer; i++ ) {
            Neuron neuron = new Neuron(
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ),
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ) );
            network.addNeuron( neuron, midLayer1 );
        }
        for( int i = 0; i < neuronsInLayer; i++ ) {
            Neuron neuron = new Neuron(
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ),
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ) );
            network.addNeuron( neuron, midLayer2 );
        }
        network.addNeuron( outN, outputLayer );
        network.addToOutputs( outN );
        //network.addOutputConsumer( outN, network::collectOutput );

        for( Neuron neuron : network.layerIterable( midLayer1 ) ) {
            network.addConnection( inN, neuron, 1.0 );
            for( Neuron neuron2 : network.layerIterable( midLayer2 ) ) {
                network.addConnection( neuron, neuron2, 2.0 / neuronsInLayer * Rng.nextDouble() );
            }
        }

        for( Neuron neuron : network.layerIterable( midLayer2 ) ) {
            network.addConnection( neuron, outN, 2.0 / neuronsInLayer * Rng.nextDouble() );
        }

        return network;
    }

    // 2-layer network population (10k)
    public static void test4 () {
        double[][] inputs = valuesLinear1;
        double[][] expectedOutputs = valuesStep1;

        int inputCount = inputs.length;
        int popCount = 10_000, bestCount = 100;

        for( Rng.rndMax = 2.0; Rng.rndMax < 4.0; Rng.rndMax += 0.1 ) {
            System.out.println( "Rng.rndMax = " + Rng.rndMax );
            ArrayList<Network> population = new ArrayList<>();
            for( int i = 0; i < popCount; i++ ) {
                population.add( createTestTwoLayerNetwork1() );
            }
            int i = 0;
            for( Network network : population ) {
                for( int j = 0; j < inputCount; j++ ) {
                    network.addInputs( inputs[j] );
                    network.process();
                    network.calcOutputDist( expectedOutputs[j] );
                    /*
                     for( double d : network.getOutputs() ) {
                     System.out.println( d );
                     }
                     */
                }
                //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
                i++;
            }
            population.sort( (Network o1, Network o2)
                    -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );

            double sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg = " + sumL2 / popCount );

            population = new ArrayList<>( population.subList( 0, bestCount ) );

            sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg_best = " + sumL2 / bestCount );
            //mix/mutate population
        }
        // expected best @ Rng.rndMax = 2.5
    }

    /*
     private static TransferFunction createRandomTransferFunction () {
     switch ( rng.nextInt( 3 ) ) {
     case 1:
     return new TransferFunction.Log(rnd(), rnd() );
     case 2:
     return new TransferFunction.Log( rnd(), rnd() );
     //return new TransferFunction.Exp( rnd(), rnd() );
     case 0:
     return new TransferFunction.Log( rnd(), rnd() );
     //return new TransferFunction.Lin( rnd(), rnd() );

     }
     throw new RuntimeException();
     }
     */
    private static TransferFunction createRandomTransferFunction () {
        switch ( Rng.nextInt( 10 ) ) {
            case 0:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 1:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 2:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 3:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 4:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 5:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 6:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 7:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Log );
            case 8:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Log );
            case 9:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Log );
                //return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Sin );
        }
        throw new RuntimeException();
    }

    private static Network createTestTwoLayerNetwork2 () {
        Network network = new Network();

        Neuron inN = new Neuron( IDENTITY, IDENTITY );
        Neuron outN = new Neuron( IDENTITY, IDENTITY );

        //int neuronsInLayer1 = 2, neuronsInLayer2 = 2;
        int foo = 2;
        int neuronsInLayer1 = foo, neuronsInLayer2 = foo;

        Layer inputLayer = new Layer();
        Layer midLayer1 = new Layer();
        Layer midLayer2 = new Layer();
        Layer outputLayer = new Layer();

        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        network.addNeuron( inN, inputLayer );
        for( int i = 0; i < neuronsInLayer1; i++ ) {
            Neuron neuron = new Neuron( createRandomTransferFunction(), createRandomTransferFunction() );
            network.addNeuron( neuron, midLayer1 );
        }
        for( int i = 0; i < neuronsInLayer2; i++ ) {
            Neuron neuron = new Neuron( createRandomTransferFunction(), createRandomTransferFunction() );
            network.addNeuron( neuron, midLayer2 );
        }
        network.addNeuron( outN, outputLayer );
        network.addToOutputs( outN );
        //network.addOutputConsumer( outN, network::collectOutput );

        for( Neuron neuron : network.layerIterable( midLayer1 ) ) {
            network.addConnection( inN, neuron, 1.0 );
            for( Neuron neuron2 : network.layerIterable( midLayer2 ) ) {
                //neuron.addConnection( neuron2, 2.0 / neuronsInLayer1 * Rng.nextDouble()  );
                network.addConnection( neuron, neuron2, 1.0 / neuronsInLayer1 );
            }
        }

        for( Neuron neuron : network.layerIterable( midLayer2 ) ) {
            //neuron.addConnection( outN, 2.0 / neuronsInLayer2 * Rng.nextDouble()  );
            network.addConnection( neuron, outN, 1.0 / neuronsInLayer2 );
        }

        return network;
    }

    public static void test5 () {
        double[][] inputs = valuesLinear1;
        double[][] expectedOutputs = valuesStep1;

        int inputCount = inputs.length;
        int popCount = 100_000, bestCount = 100;

        for( Rng.rndMax = 3.0; Rng.rndMax < 6.0; Rng.rndMax += 0.1 ) {
            System.out.println( "Rng.rndMax = " + Rng.rndMax );
            ArrayList<Network> population = new ArrayList<>();
            for( int i = 0; i < popCount; i++ ) {
                population.add( createTestTwoLayerNetwork2() );
            }
            int i = 0;
            for( Network network : population ) {
                for( int j = 0; j < inputCount; j++ ) {
                    network.addInputs( inputs[j] );
                    network.process();
                    network.calcOutputDist( expectedOutputs[j] );
                    /*
                     for( double d : network.getOutputs() ) {
                     System.out.println( d );
                     }
                     */
                }
                //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
                i++;
            }
            population.sort( (Network o1, Network o2)
                    -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );

            double sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg = " + sumL2 / popCount );

            population = new ArrayList<>( population.subList( 0, bestCount ) );

            sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg_best = " + sumL2 / bestCount );
            //mix/mutate population
        }
    }

    public static void test6 () {
        GsonBuilder gb = new GsonBuilder();
        gb.setPrettyPrinting();
        Gson gson = gb.create();

        double[][] inputs = valuesLinear2;
        double[][] expectedOutputs = valuesSquare2;

        int inputCount = inputs.length;
        int popCount = 10_000, bestCount = 100;
        int ITERATIONS = 20;

        //int xRange = 1000, yRange = 1000;
        int xRange = 500, yRange = 500;
        //BufferedImage bi = new BufferedImage( xRange, yRange, BufferedImage.TYPE_INT_ARGB );
        BufferedImage bi = new BufferedImage( xRange, yRange, BufferedImage.TYPE_INT_RGB );

        JFrame outFrame = new JFrame( "best network result" );
        JLabel imageLabel = new JLabel( new ImageIcon( bi ) );
        outFrame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );
        outFrame.add( imageLabel );
        outFrame.pack();
        outFrame.setLocationRelativeTo( null );
        outFrame.setVisible( true );

        Rng.rndMax = 3.0;
        System.out.println( "Rng.rndMax = " + Rng.rndMax );
        ArrayList<Network> population = new ArrayList<>();
        for( int i = 0; i < popCount; i++ ) {
            population.add( createTestTwoLayerNetwork2() );
        }

        int i = 0;
        for( Network network : population ) {
            for( int j = 0; j < inputCount; j++ ) {
                network.addInputs( inputs[j] );
                network.process();
                network.calcOutputDist( expectedOutputs[j] );
                /*
                 for( double d : network.getOutputs() ) {
                 System.out.println( d );
                 }
                 */
            }
            //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
            i++;
        }
        /*
         population.sort( (Network o1, Network o2)
         -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );
         */
        population.sort( (Network o1, Network o2)
                -> ( o1.getDistL1() < o2.getDistL1() ) ? -1 : ( ( o1.getDistL1() > o2.getDistL1() ) ? 1 : 0 ) );

        double sumL1 = 0, sumL2 = 0;
        i = 0;
        for( Network network : population ) {
            //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
            sumL1 += network.getDistL1();
            sumL2 += network.getDistL2();
            i++;
        }
        System.out.println( "L1avg = " + sumL2 / popCount );
        System.out.println( "L2avg = " + sumL2 / popCount );

        sumL2 = 0;
        i = 0;
        for( Network network : population.subList( 0, bestCount ) ) {
            //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
            sumL1 += network.getDistL1();
            sumL2 += network.getDistL2();
            i++;
        }
        System.out.println( "L1avg_best = " + sumL1 / bestCount );
        System.out.println( "L2avg_best = " + sumL2 / bestCount );

        for( int count = 0; count < ITERATIONS; count++ ) {
            System.out.println( "* " + count + " ****************************" );
            ArrayList<Network> newPopulation = new ArrayList<>();
            int mutations = popCount / bestCount;
            for( Network network : population.subList( 0, bestCount ) ) {
                newPopulation.add( network );
                for( int j = 0; j < mutations; j++ ) {
                    newPopulation.add( network.copy().mutate( /* Rng.rng, */ 0.025, 0.025, 0.001 ) );
                }
            }
            population = newPopulation;

            i = 0;
            for( Network network : population ) {
                for( int j = 0; j < inputCount; j++ ) {
                    network.addInputs( inputs[j] );
                    network.process();
                    network.calcOutputDist( expectedOutputs[j] );
                    /*
                     for( double d : network.getOutputs() ) {
                     System.out.println( d );
                     }
                     */
                }
                //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
                i++;
            }
            /*
             population.sort( (Network o1, Network o2)
             -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );
             */
            population.sort( (Network o1, Network o2)
                    -> ( o1.getDistL1() < o2.getDistL1() ) ? -1 : ( ( o1.getDistL1() > o2.getDistL1() ) ? 1 : 0 ) );

            sumL1 = 0;
            sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL1 += network.getDistL1();
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L1avg = " + sumL1 / popCount );
            System.out.println( "L2avg = " + sumL2 / popCount );

            sumL2 = 0;
            i = 0;
            for( Network network : population.subList( 0, bestCount ) ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL1 += network.getDistL1();
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L1avg_best = " + sumL1 / bestCount );
            System.out.println( "L2avg_best = " + sumL2 / bestCount );

            Network bestNetwork = population.get( 0 );
            double[] inputArr = { 0 };

            double minInput = -10;
            double maxInput = 10;
            double range = maxInput - minInput;
            double step = range / xRange;
            double d;

            Graphics g = bi.createGraphics();
            g.setColor( Color.WHITE );
            g.fillRect( 0, 0, xRange, yRange );
            g.setColor( Color.GRAY );
            g.drawLine( 0, yRange / 2, xRange, yRange / 2 );
            g.drawLine( xRange / 2, 0, xRange / 2, yRange );
            g.setColor( Color.CYAN );
            g.drawLine( 0, yRange * 45 / 100, xRange, yRange * 45 / 100 );
            g.drawLine( 0, yRange * 55 / 100, xRange, yRange * 55 / 100 );
            g.drawLine( xRange * 45 / 100, 0, xRange * 45 / 100, yRange );
            g.drawLine( xRange * 55 / 100, 0, xRange * 55 / 100, yRange );

            for( int counter = 0; counter < xRange; counter++ ) {
                d = Math.round( ( minInput + step * counter + 0.01 ) * 100 ) / 100.0;
                inputArr[0] = d;
                bestNetwork.addInputs( inputArr );
                bestNetwork.process();
                double output = Math.round( bestNetwork.getOutputs()[0] * 1000 ) / 1000.0;
                //out.println( d + "," + output );
                int y = yRange - (int) Misc.clamp( ( output - minInput ) / step, 1, yRange );
                bi.setRGB( counter, y, Color.BLACK.getRGB() );
            }
            imageLabel.repaint();
        }
        /*
         population.sort( (Network o1, Network o2)
         -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );
         */
        Network bestNetwork = population.get( 0 );

        try ( PrintWriter out = new PrintWriter( "output/bestANN.json" ); ) {
            out.print( gson.toJson( bestNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        double[] inputArr = { 0 };
        try ( PrintWriter out = new PrintWriter( "output/wideBestResults.txt" ); ) {
            //out.println( "IN,OUT" );
            out.println( "data = {" );
            //for( double d = -10; d < 10; d = Math.round( ( d + 0.01 ) * 100 ) / 100.0 ) {
            double minInput = -10;
            double maxInput = 10;
            double range = maxInput - minInput;
            double step = range / xRange;
            double d;

            Graphics g = bi.createGraphics();
            g.setColor( Color.WHITE );
            g.fillRect( 0, 0, xRange, yRange );
            g.setColor( Color.GRAY );
            g.drawLine( 0, yRange / 2, xRange, yRange / 2 );
            g.drawLine( xRange / 2, 0, xRange / 2, yRange );
            g.setColor( Color.CYAN );
            g.drawLine( 0, yRange * 45 / 100, xRange, yRange * 45 / 100 );
            g.drawLine( 0, yRange * 55 / 100, xRange, yRange * 55 / 100 );
            g.drawLine( xRange * 45 / 100, 0, xRange * 45 / 100, yRange );
            g.drawLine( xRange * 55 / 100, 0, xRange * 55 / 100, yRange );

            for( int counter = 0; counter < xRange; counter++ ) {
                d = Math.round( ( minInput + step * counter + 0.01 ) * 100 ) / 100.0;
                inputArr[0] = d;
                bestNetwork.addInputs( inputArr );
                bestNetwork.process();
                double output = Math.round( bestNetwork.getOutputs()[0] * 1000 ) / 1000.0;
                //out.println( d + "," + output );
                int y = yRange - (int) Misc.clamp( ( output - minInput ) / step, 1, yRange );
                bi.setRGB( counter, y, Color.BLACK.getRGB() );
                out.println( "{" + d + "," + output + "}," );
            }
            imageLabel.repaint();
            out.println( "};" );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        Network worstNetwork = population.get( population.size() - 1 );

        try ( PrintWriter out = new PrintWriter( "output/worstANN.json" ); ) {
            out.print( gson.toJson( worstNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        try ( PrintWriter out = new PrintWriter( "output/wideWorstResults.txt" ); ) {
            //out.println( "IN,OUT" );
            out.println( "data = {" );
            for( double d = -10; d < 10; d += 0.01 ) {
                inputArr[0] = d;
                worstNetwork.addInputs( inputArr );
                worstNetwork.process();
                double output = worstNetwork.getOutputs()[0];
                //out.println( d + "," + output );
                out.println( "{" + d + "," + output + "}," );
            }
            out.println( "};" );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }
    }

}
