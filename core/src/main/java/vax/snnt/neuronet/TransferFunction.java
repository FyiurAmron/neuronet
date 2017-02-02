package vax.snnt.neuronet;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.function.DoubleUnaryOperator;

/**

 @author toor
 */
public /* abstract */ class TransferFunction implements DoubleUnaryOperator {
    protected double ampIn, ampOut, shiftIn, shiftOut;
    protected final FunctionType functionType;
    protected final transient DoubleUnaryOperator operator;

    public static enum FunctionType {
        Lin( x -> x ),
        Log( x -> ( x > 0 ) ? log( x + 1 ) : -log( -x + 1 ) ),
        Exp( x -> ( x > 0 ) ? exp( x ) - 1 : -exp( -x ) + 1 ),
        Sin( x -> Math.sin( x ) ), //
        ;
        private final DoubleUnaryOperator lambda;

        private FunctionType ( DoubleUnaryOperator lambda ) {
            this.lambda = lambda;
        }

        public DoubleUnaryOperator getLambda () {
            return lambda;
        }
    }

    public static TransferFunction IDENTITY = new TransferFunction( 1, 1, 0, 0, FunctionType.Lin ) {
        @Override
        public double applyAsDouble ( double x ) {
            return x;
        }
    };

    public TransferFunction ( double ampIn, double ampOut, double shiftIn, double shiftOut, FunctionType functionType ) {
        this.ampIn = ampIn;
        this.ampOut = ampOut;
        this.shiftIn = shiftIn;
        this.shiftOut = shiftOut;
        this.functionType = functionType;
        operator = functionType.getLambda();
    }

    private TransferFunction ( double ampIn, double ampOut, double shiftIn, double shiftOut, DoubleUnaryOperator operator ) {
        this.ampIn = ampIn;
        this.ampOut = ampOut;
        this.shiftIn = shiftIn;
        this.shiftOut = shiftOut;
        functionType = null;
        this.operator = operator;
    }

    private TransferFunction ( double ampIn, double ampOut, double shiftIn, double shiftOut,
            FunctionType functionType, DoubleUnaryOperator operator ) {
        this.ampIn = ampIn;
        this.ampOut = ampOut;
        this.shiftIn = shiftIn;
        this.shiftOut = shiftOut;
        this.functionType = functionType;
        this.operator = operator;
    }

    public TransferFunction ( TransferFunction transferFunction ) {
        this( transferFunction.ampIn, transferFunction.ampOut, transferFunction.shiftIn, transferFunction.shiftOut,
                transferFunction.functionType, transferFunction.operator );
    }

    public TransferFunction copy () {
        return new TransferFunction( this );
    }

    @Override
    public double applyAsDouble ( double x ) {
        return ampOut * operator.applyAsDouble( ampIn * x + shiftIn ) + shiftOut;
    }

    public double f ( double x ) {
        return applyAsDouble( x );
    }

    @Override
    public String toString () {
        return ( functionType == null ) ? "?" : "" + functionType;
    }
}
