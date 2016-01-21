#ifndef STORM_PARSER_SPIRITPARSERDEFINITIONS_H_
#define	STORM_PARSER_SPIRITPARSERDEFINITIONS_H_

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-#pragma-messages"

// Include boost spirit.
#define BOOST_SPIRIT_USE_PHOENIX_V3
#include <boost/typeof/typeof.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>
#include <boost/spirit/home/classic/iterator/position_iterator.hpp>

#pragma clang diagnostic pop

namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;

typedef std::string::const_iterator BaseIteratorType;
typedef boost::spirit::line_pos_iterator<BaseIteratorType> PositionIteratorType;
typedef PositionIteratorType Iterator;

typedef BOOST_TYPEOF(boost::spirit::ascii::space | qi::lit("//") >> *(qi::char_ - (qi::eol | qi::eoi)) >> (qi::eol | qi::eoi)) Skipper;

#endif /* STORM_PARSER_SPIRITPARSERDEFINITIONS_H_ */